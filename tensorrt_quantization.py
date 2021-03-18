import os
import numpy as np
from sklearn.metrics import classification_report
import torch

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

TRT_LOGGER = trt.Logger()
#TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def GiB(val):
    return val * 1 << 30

class BatchStream():
	def __init__(self, batch_size, dataloader):
		self.batch_size = batch_size
		self.dataloader = iter(dataloader)
		self.batch_id = 0
	def next_batch(self):
		self.batch_id += 1

		data, labels = next(self.dataloader)

		data_np = data.detach().cpu().numpy()
		return np.ascontiguousarray(data_np, dtype=np.float32)

class MyCalibrator(trt.IInt8EntropyCalibrator):
	def __init__(self, batchstream, shape, cache_file='int8_calibration.cache'):
		trt.IInt8EntropyCalibrator.__init__(self)

		self._batch_size = shape[0]
		self.cache_file = cache_file
		self.batchstream = batchstream

		#device_size = cuda.pagelocked_empty(shape, dtype=trt.nptype(trt.DataType.FLOAT)).nbytes
		#nbytes = np.zeros(shape=(32, 320)).nbytes
		nbytes = int(np.prod(shape) * trt.float32.itemsize)
		self.device_input = cuda.mem_alloc(nbytes)

	def get_batch(self, names, p_str=None):
		try:
			batch = self.batchstream.next_batch()
			cuda.memcpy_htod(self.device_input, batch)
			return [int(self.device_input)]
		except StopIteration:
			return None

	def get_batch_size(self):
		return self._batch_size

	def read_calibration_cache(self):
		if os.path.exists(self.cache_file):
			with open(self.cache_file, "rb") as f:
				return f.read()

	def write_calibration_cache(self, cache):
		with open(self.cache_file, "wb") as f:
			f.write(cache)



def build_engine(onnx_file_path, method, dataloader=None):

	# initialize TensorRT engine and parse ONNX model
	builder = trt.Builder(TRT_LOGGER)
	config = builder.create_builder_config()

	network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
	network = builder.create_network(network_flags)
	parser = trt.OnnxParser(network, TRT_LOGGER)

	# parse ONNX
	with open(onnx_file_path, 'rb') as model:
		print('Beginning ONNX file parsing')
		parser.parse(model.read())
	print('Completed parsing of ONNX file')
	
	# allow TensorRT to use up to 1GB of GPU memory for tactic selection
	builder.max_workspace_size = GiB(1) #1 << 30
	builder.max_batch_size = 32
	
	if method == 0:
		if builder.platform_has_fast_fp16:
			print('FP16 mode enabled')
			builder.fp16_mode = True
		else:
			print('No float16 support.')
	elif method == 1:
		if builder.platform_has_fast_int8:
			
			for data, label in dataloader:
				shape = data.shape
				break
			
			batchstream = BatchStream(shape[0], dataloader)

			builder.int8_mode = True
			builder.int8_calibrator = MyCalibrator(batchstream, shape)
		else:
			print('No Int8 support.')

	# generate TensorRT engine optimized for the target platform
	print('Building an engine...')
	engine = builder.build_cuda_engine(network)
	#engine = builder.build_engine(network, config)
	context = engine.create_execution_context()
	print("Completed creating Engine")

	return engine, context

def allocate(engine, context):

	#print('Number of bindings:', engine.num_bindings)
	
	# Calculate binding size
	input_size = trt.volume(engine.get_binding_shape(0)) #* engine.max_batch_size
	input_dtype = trt.nptype(engine.get_binding_dtype(0))
	#print('Input binding shape:', engine.get_binding_shape(0), 'Input dtype:', engine.get_binding_dtype(0))
	
	output_size = trt.volume(engine.get_binding_shape(1)) #* engine.max_batch_size
	output_dtype = trt.nptype(engine.get_binding_dtype(1))
	
	#Allocate memory required for input data and for output data
	host_input = cuda.pagelocked_empty(input_size, input_dtype)
	host_output = cuda.pagelocked_empty(output_size, output_dtype)
			
	device_input = cuda.mem_alloc(host_input.nbytes)
	device_output = cuda.mem_alloc(host_output.nbytes)
	
	return host_input, host_output, device_input, device_output



def tensorrt_compression(model, dataloader, params):

	# Get dataset (sizes and data)
	full_data = torch.empty(0)
	full_labels = torch.empty(0)
	batch_size = None
	for data, labels in dataloader:
		if batch_size != None and data.shape[0] == batch_size:
			full_data = torch.cat((full_data, data))
			full_labels = torch.cat((full_labels, labels))
		if batch_size is None:
			batch_size = data.shape[0]
		

	#full_labels = full_labels[:1024]
	#full_data =  full_data[:1024]

	len_data = full_labels.shape[0]
	nr_classes = int(torch.max(full_labels)+1)
	full_data = full_data.to('cuda')
	print(full_data.shape)

	# Export to ONNX
	if params['method'] == 0:

		# Export dynamic batch model
		mname_dyn_batch = 'model_dynbatch.onnx'
		torch.onnx.export(model, full_data, mname_dyn_batch, input_names=['input'], 
		                  output_names=['output'], export_params=True, opset_version=12)

	elif params['method'] == 1:
		# Export fixed batch model
		mname_fixed_batch = 'model_fixedbatch.onnx'
		torch.onnx.export(model, full_data[:batch_size], mname_fixed_batch, input_names=['input'], 
		                  output_names=['output'], export_params=True)
		
		# Export dynamic batch model
		mname_dyn_batch = 'model_dynbatch.onnx'
		torch.onnx.export(model, full_data, mname_dyn_batch, input_names=['input'], 
		                  output_names=['output'], export_params=True) # dynamic_axes={'input':{0: 'batch_size'}}
		# Build engine for calibration
		engine2, context2 = build_engine(mname_fixed_batch, method=params['method'], dataloader=dataloader)
	

	
	# Build engine 
	engine, context = build_engine(mname_dyn_batch, method=params['method'], dataloader=dataloader)

	# Allocate space 
	host_input, host_output, device_input, device_output = allocate(engine, context)

	#y_preds_all = []
	# ----------

	stream = cuda.Stream()
	full_data = full_data.to('cpu')
	#data = data.detach().cpu().numpy()
	np.copyto(host_input, full_data.flatten())
	#np.copyto(host_input, full_data.flatten())
	#host_input = np.array(test_data_flat, dtype=np.float32, order='C')
	cuda.memcpy_htod(device_input, host_input)

	# Run inference
	context.execute(bindings=[int(device_input), int(device_output)], batch_size=32)
	
	# Copy results from cuda
	cuda.memcpy_dtoh(host_output, device_output)
	
	# Run evaluation	
	y_preds = np.zeros(shape=(len_data*nr_classes)) 
	np.copyto(y_preds, host_output)

	y_preds = np.resize(y_preds, (len_data, nr_classes))
	y_preds = np.argmax(y_preds, axis=1)

	jsonres = classification_report(full_labels, y_preds, output_dict=True)
	return jsonres



'''
# Old, batched version


	y_preds = []
	i = 0
	for data, labels in dataloader:

		print('Batch {} of {}'.format(i, int(len_data/batch_size)), end='\r')
		i=i+1
		if data.shape[0] != batch_size:
			break

		# Copy data to cuda
		stream = cuda.Stream()
		#full_data = full_data.to('cpu')
		data = data.detach().cpu().numpy()
		np.copyto(host_input, data.flatten())
		#np.copyto(host_input, full_data.flatten())
		#host_input = np.array(test_data_flat, dtype=np.float32, order='C')
		cuda.memcpy_htod(device_input, host_input)

		# Run inference
		context.execute_v2(bindings=[int(device_input), int(device_output)])
		
		# Copy results from cuda
		cuda.memcpy_dtoh(host_output, device_output)
		stream.synchronize()
		
		# Run evaluation	
		y_preds_batch = np.zeros(shape=(batch_size*nr_classes)) 
		#y_preds_batch = []
		np.copyto(y_preds_batch, host_output)

		#y_preds = np.concatenate(y_preds, y_preds_batch)
		y_preds += y_preds_batch.tolist()


'''


'''
	stream = cuda.Stream()
	full_data = full_data.to('cpu')
	#data = data.detach().cpu().numpy()
	np.copyto(host_input, full_data.flatten())
	#np.copyto(host_input, full_data.flatten())
	#host_input = np.array(test_data_flat, dtype=np.float32, order='C')
	cuda.memcpy_htod(device_input, host_input)

	# Run inference
	context.execute(bindings=[int(device_input), int(device_output)], batch_size=32)
	
	# Copy results from cuda
	cuda.memcpy_dtoh(host_output, device_output)
	stream.synchronize()
	
	# Run evaluation	
	y_preds = np.zeros(shape=(len_data*nr_classes)) 
	np.copyto(y_preds, host_output)

	y_preds = np.resize(y_preds, (len_data, nr_classes))
	y_preds = np.argmax(y_preds, axis=1)

'''