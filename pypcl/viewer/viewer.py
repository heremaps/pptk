import subprocess
import struct
import socket
import numpy
import os
import inspect
import warnings

_viewer_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
_viewer_dir = os.path.abspath(_viewer_dir) if ~os.path.isabs(_viewer_dir) else _viewed_dir
__all__ = ['viewer']
class viewer:
	def __init__(self,*args,**kwargs):
		# ensure positions is 3-column array of float32s
		positions = numpy.asarray(args[0],dtype=numpy.float32).reshape(-1,3)
		attr = args[1:]
		color_map = kwargs.get('color_map','jet')
		scale = kwargs.get('scale',None)
		debug = kwargs.get('debug',False)
		
		# start up viewer in separate process
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s.bind(('localhost',0))
		s.listen(0)
		self._process = subprocess.Popen(\
			[os.path.join(_viewer_dir,'viewer'),str(s.getsockname()[1])],
			stdout = subprocess.PIPE,
			stderr = None if debug else subprocess.PIPE);
		if debug: print 'Started viewer process: %s'%os.path.join(_viewer_dir,'viewer')
		x = s.accept()
		self._portNumber = struct.unpack('H',self._process.stdout.read(2))[0]
		#self._portNumber = struct.unpack('H',x[0].recv(2))[0]
		
		# upload points to viewer
		self.__load(positions)
		self.attributes(*attr)
		self.color_map(color_map,scale)
		
	def close(self):
		self._process.kill()
		pass

	def clear(self):
		# construct message
		msg = struct.pack('b',2)
		# send message to viewer
		self.__send(msg)

	def reset(self):
		# construct message
		msg = struct.pack('b',3)
		# send message to viewer
		self.__send(msg)
	
	def set(self,**kwargs):
		for prop,val in kwargs.items():
			self.__send(_construct_set_msg(prop,val))
	
	def get(self,prop_name):
		return self.__query(_construct_get_msg(prop_name))
		
	def load(self,*args,**kwargs):
		positions = numpy.asarray(args[0],dtype=numpy.float32).reshape(-1,3)
		attr = args[1:]
		color_map = kwargs.get('color_map','jet')
		scale = kwargs.get('scale',None)
		self.__load(positions)
		self.attributes(*attr)
		self.color_map(color_map,scale)

	def attributes(self,*attr):
		msg = struct.pack('Q',len(attr))
		error_msg = '%d-th attribute array inconsistent with number of points'
		for i,x in enumerate(attr):
			x = numpy.asarray(x,dtype=numpy.float32)
			# TODO:warn if attribute array contains NaN
			# array of scalars
			if len(x.shape)==1:
				if x.shape[0]!=self.get('num_points') and x.shape[0]!=1:
					raise ValueError(error_msg%i)
				msg += struct.pack('QQ',x.shape[0],1)+x.tostring()				
			# array of rgb or rgba
			elif len(x.shape)==2 and (x.shape[-1]==4 or x.shape[-1]==3):
				if x.shape[0]!=self.get('num_points') and x.shape[0]!=1:
					raise ValueError(error_msg%i)
				if x.shape[-1]==3:
					x = numpy.c_[x,numpy.ones(x.shape[0],dtype=numpy.float32)]
				msg += struct.pack('QQ',*x.shape)+x.tostring()
			else:
				raise ValueError('%d-th attribute array shape is not supported'%i)
		msg = struct.pack('b',10)+struct.pack('Q',len(msg))+msg
		self.__send(msg)
	
	def color_map(self,c,scale=None):
		# accepts array of rgb or rgba vectors
		if isinstance(c,str):
			c = _color_maps[c]
		elif isinstance(c,list):
			c = numpy.array(c)
		if len(c.shape)!=2 or c.shape[1]!=3 and c.shape[1]!=4:
			raise ValueError('Expecting array of rgb/rgba vectors')
		if c.shape[1]==3:
			c = numpy.c_[c,numpy.ones(c.shape[0])]
		self.set(color_map=c)
		if scale is None:
			self.set(color_map_scale=[0,0])
		else:
			self.set(color_map_scale=scale)
		
	def capture(self,filename):
		msg = struct.pack('b',6)+\
			_pack_string(os.path.abspath(filename))
		self.__send(msg)

	def play(self,poses,ts=[],tlim=[-numpy.inf,numpy.inf],repeat=False,interp='cubic_natural'):
		poses,ts = _fix_poses_ts_input(poses,ts)
		if poses.size==0: return
		msg = struct.pack('b',8)+\
			struct.pack('i',poses.shape[0])+poses.tostring()+\
			struct.pack('i',ts.size)+ts.tostring()+\
			struct.pack('b',_interp_code[interp])
		self.__send(msg)
		msg = struct.pack('b',9)+\
			struct.pack('2f',*tlim)+\
			struct.pack('?',repeat)
		self.__send(msg)
	
	def record(self,folder,poses,ts=[],\
			tlim=[-numpy.inf,numpy.inf],interp='cubic_natural',\
			shutter_speed=numpy.inf,fps=24,prefix='frame_',ext='png'):
		# note: generate video from resulting image sequence, for example:
		# >> ffmpeg -i "frame_%03d.png" -c:v mpeg4 -qscale:v 0 -r 24 output.mp4
		if not os.path.isdir(folder):
			raise ValueError('invalid folder provided')
		poses,ts = _fix_poses_ts_input(poses,ts)
		if poses.size==0: return
		# load camera path
		msg = struct.pack('b',8)+\
			struct.pack('i',poses.shape[0])+poses.tostring()+\
			struct.pack('i',ts.size)+ts.tostring()+\
			struct.pack('b',_interp_code[interp])
		self.__send(msg)
		
		# clamp tlim[0] and tlim[1] to [ts[0],ts[-1]]
		t_beg = numpy.minimum(numpy.maximum(ts[0],tlim[0]),ts[-1])
		t_end = numpy.minimum(numpy.maximum(ts[0],tlim[1]),ts[-1])

		# ensure t_beg <= t_end
		t_end = numpy.maximum(t_end,t_beg)
		
		# pose and capture
		num_frames = 1+numpy.floor((t_end-t_beg)*fps)
		num_digits = 1+numpy.floor(numpy.log10(num_frames))
		for i in range(int(num_frames)):
			t = i*1.0/fps+t_beg
			msg = struct.pack('b',9)+\
				struct.pack('2f',t,t)+\
				struct.pack('?',False)
			self.__send(msg)
			filename = prefix+('%0'+str(num_digits)+'d')%(i+1)+'.'+ext
			filename = os.path.join(folder,filename)
			self.capture(filename)
			# todo: need to check whether write succeeded
			#       ideally, capture(...) should return filename
		
	def wait(self):
		s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		s.connect(('localhost',self._portNumber))
		s.send(struct.pack('b',7))
		s.setblocking(1)
		buf = ''
		while len(buf) == 0:
			buf = s.recv(1)
		if buf != 'x':
			raise RuntimeError('expecting return code \'x\'')
		s.close()
		
	def __load(self,positions):
		# if no points, then done
		if positions.size == 0:
			return
		# construct message
		numPoints = positions.size / 3
		msg = struct.pack('b',1)+\
			struct.pack('i',numPoints)+\
			positions.tostring()
		# send message to viewer
		self.__send(msg)
	
	def __send(self,msg):
		s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		s.connect(('localhost',self._portNumber))
		totalSent = 0
		while totalSent < len(msg):
			sent = s.send(msg)
			if sent == 0:
				raise RuntimeError("socket connection broken")
			totalSent = totalSent + sent
		s.close()

	def __query(self,msg):
		s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		#s.setsockopt(socket.SOL_SOCKET,socket.TCP_NODELAY,1)
		s.connect(('localhost',self._portNumber))
		totalSent = 0
		while totalSent < len(msg):
			sent = s.send(msg)
			if sent == 0:
				raise RuntimeError("socket connection broken")
			totalSent = totalSent + sent
		# layout of response message:
		# 0: data type (0 - error msg, 1 - char, 2 - float, 3 - int, 4 - uint)
		# 1: number of dimensions (quint64)
		# 9: dimensions (quint64)
		# ?: body
		lookupSize = {0:1,1:1,2:4,3:4,4:4}
		dataType = ord(s.recv(1))
		numDims = struct.unpack('Q',_recv_from_socket(8,s))[0]
		dims = struct.unpack(str(numDims)+'Q',_recv_from_socket(numDims*8,s))
		numElts = numpy.prod(dims)
		bodySize = lookupSize[dataType]*numElts
		body = _recv_from_socket(bodySize,s)
		s.close()

		if dataType == 0:
			raise ValueError(body)
		if dataType != 0 and dataType != 1:
			lookupCode = {1:'c',2:'f',3:'i',4:'I'}
			body = list(struct.unpack(str(numElts)+lookupCode[dataType],body))
			body = numpy.array(body).reshape(dims)
		# return body as is if type is char (0)
		return body

def _recv_from_socket(n,s):
	# receive n bytes from socket s
	buf = ''
	while len(buf) < n:
		buf += s.recv(n-len(buf))
	return buf

def _fix_poses_ts_input(poses,ts):
	# ensure poses is 6 column array of floats
	poses = numpy.float32(numpy.array(poses).reshape(-1,6)).copy()
	
	# ensure ts has the same number of timestamps as poses
	ts = numpy.float32(numpy.array(ts))
	if ts.size==0:
		ts = numpy.float32(numpy.arange(poses.shape[0]))
	elif ts.size!=poses.shape[0]:
		raise ValueError('number of time stamps != number of key poses')

	# ensure ts is unique and ascending
	if numpy.any(numpy.diff(ts)<=0):
		raise ValueError('time stamps must be unique and ascending')
	
	# ensure subsequent angle differences between -180 and +180 degrees
	def correct_angles(x):
		# note: mapping takes +180 + 360k to +180, 
		#       and -180 + 360k to -180, for any integer k
		d = numpy.diff(x)
		absd = numpy.abs(d)
		y = -absd-2.0*numpy.pi*numpy.floor((-absd+numpy.pi)/2.0/numpy.pi)
		y *= -numpy.sign(d)
		return x[0]+numpy.r_[0,numpy.cumsum(y)]
	poses[:,3] = correct_angles(poses[:,3])
	poses[:,4] = correct_angles(poses[:,4])
		
	return (poses,ts)
	
def _encode_bool(x):
	try: y = struct.pack('?',x)
	except: raise
	return y
	
def _encode_float(x):
	try: y = struct.pack('f',x)
	except: raise
	return y

def _encode_floats(x):
	return numpy.asarray(x,dtype=numpy.float32).tostring()

def _encode_uints(x):
	return numpy.asarray(numpy.uint32(x)).tostring()

def _encode_uint(x):
	try: y = struct.pack('I',x)
	except: raise
	return y
	
def _encode_rgb(x):
	x = numpy.asarray(numpy.float32(x))
	if x.size!=3 or \
		numpy.any(numpy.logical_or(x<0.0,x>1.0)):
		raise ValueError('Expecting 3 values in [0,1]')
	return struct.pack('fff',x[0],x[1],x[2])

def _encode_rgba(x):
	x = numpy.asarray(numpy.float32(x))
	if x.size!=4 or \
		numpy.any(numpy.logical_or(x<0.0,x>1.0)):
		raise ValueError('Expecting 4 values in [0,1]')
	return struct.pack('ffff',x[0],x[1],x[2],x[3])

def _encode_rgbas(x):
	x = numpy.asarray(numpy.float32(x))
	if x.shape[1]==4 and numpy.all(numpy.logical_and(x>=0.0,x<=1.0)):
		return x.tostring()
	else:
		raise ValueError('Expecting 4 column array of values in [0,1]')
		
def _encode_xyz(x):
	x = numpy.asarray(numpy.float32(x))
	if x.size!=3:
		raise ValueError('Expecting 3 values')
	return struct.pack('fff',x[0],x[1],x[2])
	
def _init_properties():
	_properties['point_size'] = _encode_float
	_properties['bg_color'] = _encode_rgba
	_properties['bg_color_top'] = _encode_rgba
	_properties['bg_color_bottom'] = _encode_rgba
	_properties['show_grid'] = _encode_bool
	_properties['show_info'] = _encode_bool
	_properties['show_axis'] = _encode_bool
	_properties['floor_level'] = _encode_float
	_properties['floor_color'] = _encode_rgba
	_properties['floor_grid_color'] = _encode_rgba
	_properties['lookat'] = _encode_xyz
	_properties['phi'] = _encode_float
	_properties['theta'] = _encode_float
	_properties['r'] = _encode_float
	_properties['selected'] = _encode_uints
	_properties['color_map'] = _encode_rgbas
	_properties['color_map_scale'] = _encode_floats
	_properties['curr_attribute_id'] = _encode_uint
	
def _construct_get_msg(prop_name):
	return struct.pack('b',5)+\
		_pack_string(prop_name)

def _construct_set_msg(prop_name,prop_value):
	if not _properties.get(prop_name):
		raise ValueError('Invalid property name encountered: %s' % prop_name)
	msg_header = struct.pack('b',4)+\
		_pack_string(prop_name)
	msg_payload = ''
	try:
		msg_payload = _properties[prop_name](prop_value)
	except BaseException as e:
		raise ValueError('Failed setting "%s": ' % prop_name + str(e))
	return msg_header + struct.pack('Q',len(msg_payload)) + msg_payload

def _pack_string(string):
	return struct.pack('Q',len(string))+\
		struct.pack(str(len(string))+'s',string)
		
def _init_color_maps():
	_color_maps['jet'] = numpy.array([[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]],dtype=numpy.float32)
	_color_maps['hsv'] = numpy.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0]],dtype=numpy.float32)
	_color_maps['hot'] = numpy.array([[0,0,0],[1,0,0],[1,1,0],[1,1,1]],dtype=numpy.float32)
	_color_maps['cool'] = numpy.array([[0,1,1],[1,0,1]],dtype=numpy.float32)
	_color_maps['spring'] = numpy.array([[1,0,1],[1,1,0]],dtype=numpy.float32)
	_color_maps['summer'] = numpy.array([[0,.5,.4],[1,1,.4]],dtype=numpy.float32)
	_color_maps['autumn'] = numpy.array([[1,0,0],[1,1,0]],dtype=numpy.float32)
	_color_maps['winter'] = numpy.array([[0,0,1],[0,1,.5]],dtype=numpy.float32)
	_color_maps['gray'] = numpy.array([[0,0,0],[1,1,1]],dtype=numpy.float32)

_properties = dict()
_init_properties()
_color_maps = dict()
_init_color_maps()
		
# define codes for each interpolation scheme
_interp_code = {'constant':0,'linear':1,'cubic_natural':2,'cubic_periodic':3}
