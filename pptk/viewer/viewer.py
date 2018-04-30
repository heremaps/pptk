import subprocess
import struct
import socket
import numpy
import os
import inspect
import warnings

_viewer_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
if ~os.path.isabs(_viewer_dir):
    _viewer_dir = os.path.abspath(_viewer_dir)

__all__ = ['viewer']


class viewer:
    def __init__(self, *args, **kwargs):
        """ Opens a point cloud viewer

        Examples:
            Create 100 random points

            >>> xyz = pptk.rand(100, 3)

            Visualize the points

            >>> pptk.viewer(xyz)

            Visualize points shaded by height

            >>> pptk.viewer(xyz, xyz[:, 2])

            Visualize points shaded by random RGB color

            >>> rgb = pptk.rand(100, 3)
            >>> pptk.viewer(xyz, rgb)

        """
        # ensure positions is 3-column array of float32s
        positions = numpy.asarray(args[0], dtype=numpy.float32).reshape(-1, 3)
        attr = args[1:]
        color_map = kwargs.get('color_map', 'jet')
        scale = kwargs.get('scale', None)
        debug = kwargs.get('debug', False)

        # start up viewer in separate process
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('localhost', 0))
        s.listen(0)
        self._process = subprocess.Popen(
            [os.path.join(_viewer_dir, 'viewer'), str(s.getsockname()[1])],
            stdout=subprocess.PIPE,
            stderr=(None if debug else subprocess.PIPE))
        if debug:
            print ('Started viewer process: %s' \
                % os.path.join(_viewer_dir, 'viewer'))
        x = s.accept()
        self._portNumber = struct.unpack('H', self._process.stdout.read(2))[0]
        # self._portNumber = struct.unpack('H',x[0].recv(2))[0]

        # upload points to viewer
        self.__load(positions)
        self.attributes(*attr)
        self.color_map(color_map, scale)

    def close(self):
        """ Closes the point cloud viewer

        Examples:

            >>> v.close()

        """
        self._process.kill()
        pass

    def clear(self):
        """ Removes the current point cloud in the viewer

        Examples:

            >>> v.clear()

        """
        # construct message
        msg = struct.pack('b', 2)
        # send message to viewer
        self.__send(msg)

    def reset(self):
        """ Resets the viewer
        """
        # construct message
        msg = struct.pack('b', 3)
        # send message to viewer
        self.__send(msg)

    def set(self, **kwargs):
        """ Sets viewer property

        =================  ===============  =================================
        Property Name      Value Type         Description
        =================  ===============  =================================
        bg_color           4 x float32      Background color in RGBA [0, 1]
        bg_color_top       4 x float32      Top background color
        bg_color_bottom    4 x float32      Bottom background color
        color_map          ? x 4 x float32  Color map; array of RGBA's [0, 1]
        color_map_scale    2 x float32      Color map scaling interval
        curr_attribute_id  uint             Current attribute set index
        floor_level        float32          Floor z-level
        floor_color        4 x float32      Floor color in RGBA [0, 1]
        lookat             3 x float32      Camera look-at position
        phi                float32          Camera azimuthal angle (radians)
        point_size         float32          Point size in world space
        r                  float32          Camera distance to look-at point
        selected           ? x uint         Indices of selected points
        show_grid          bool             Show floor grid
        show_info          bool             Show information text overlay
        show_axis          bool             Show axis / look-at cursor
        theta              float32          Camera elevation angle (radians)
        =================  ===============  =================================

        (phi, theta, r) are spherical coordinates specifying camera position
        relative to the look at position.

        (right, up, view) are orthogonal vectors forming the camera coordinate
        frame, where view is pointed away from the look at position, and view
        is the cross product of up with right

        Examples:

            >>> v = pptk.viewer(xyz)
            >>> v.set(point_size = 0.01)

        """
        for prop, val in kwargs.items():
            self.__send(_construct_set_msg(prop, val))

    def get(self, prop_name):
        """ Gets viewer property

        ================  =============  ================================
        Property Name     Return Type    Description
        ================  =============  ================================
        curr_atribute_id  uint           Current attribute set index
        eye               3 x float64    Camera position
        lookat            3 x float64    Camera look-at position
        mvp               4 x 4 float64
        num_points        uint           Number of points loaded
        num_attributes    uint           Number of attribute sets loaded
        phi               float64        Camera azimuthal angle (radians)
        r                 float64        Camera distance to look-at point
        right             3 x float64    Camera Right vector
        selected          ? x int32      Indices of selected points
        theta             float64        Camera elevation angle (radians)
        up                3 x float64    Camera up vector
        view              3 x float64    Camera view vector
        ================  =============  ================================

        Examples:

            >>> v = pptk.viewer(xyz)
            >>> v.get('selected')

        """
        return self.__query(_construct_get_msg(prop_name))

    def load(self, *args, **kwargs):
        positions = numpy.asarray(args[0], dtype=numpy.float32).reshape(-1, 3)
        attr = args[1:]
        color_map = kwargs.get('color_map', 'jet')
        scale = kwargs.get('scale', None)
        self.__load(positions)
        self.attributes(*attr)
        self.color_map(color_map, scale)

    def attributes(self, *attr):
        """ Loads point attributes

        The loaded attributes are used to clor the currently loaded point
        cloud.  Supposing n points loaded, this function accepts attributes of
        the following forms:

        * scalars: 1-d array of length 1 or n
        * RGB colors: 2-d array of shape (1, 3) or (n, 3)
        * RGBA colors: 2-d array of shape (1, 4) or (n, 4)

        Passing in no arguments clears all existing attribute sets and colors
        all points white.  Cycle through attribute sets via '[' and ']' keys.

        Examples:

        >>> xyz = pptk.rand(100, 3)
        >>> v = pptk.viewer(xyz)
        >>> attr1 = pptk.rand(100)     # 100 random scalars
        >>> attr2 = pptk.rand(100, 3)  # 100 random RGB colors
        >>> attr3 = pptk.rand(100, 4)  # 100 random RGBA colors
        >>> attr4 = pptk.rand(1, 1)    # 1 random scalar
        >>> attr5 = pptk.rand(1, 3)    # 1 random RGB color
        >>> attr6 = pptk.rand(1, 4)    # 1 random RGBA color
        >>> v.attributes(attr1, attr2, attr3, attr4, attr6)

        """
        msg = struct.pack('Q', len(attr))
        error_msg = '%d-th attribute array inconsistent with number of points'
        for i, x in enumerate(attr):
            x = numpy.asarray(x, dtype=numpy.float32)
            # TODO:warn if attribute array contains NaN
            # array of scalars
            if len(x.shape) == 1:
                if x.shape[0] != self.get('num_points') and x.shape[0] != 1:
                    raise ValueError(error_msg % i)
                msg += struct.pack('QQ', x.shape[0], 1) + x.tostring()
            # array of rgb or rgba
            elif len(x.shape) == 2 and (x.shape[-1] == 4 or x.shape[-1] == 3):
                if x.shape[0] != self.get('num_points') and x.shape[0] != 1:
                    raise ValueError(error_msg % i)
                if x.shape[-1] == 3:
                    x = numpy.c_[x,
                                 numpy.ones(x.shape[0], dtype=numpy.float32)]
                msg += struct.pack('QQ', * x.shape) + x.tostring()
            else:
                raise ValueError('%d-th ' % i +
                                 'attribute array shape is not supported')
        msg = struct.pack('b', 10) + struct.pack('Q', len(msg)) + msg
        self.__send(msg)

    def color_map(self, c, scale=None):
        """

        Specifies how scalar attributes are used to color points in the viewer.

        Input c is expected to be an array of n RGB (or RGBA) vectors
        (i.e. c is a n x 3 or n x 4 numpy array).
        Upon return, scalar values equal to scale 0 are colored with c[0],
        scale[1] with c[-1], and scalars in between appropriately interpolated.
        If scale is None, scale is automatically set as the minimum and maximum
        scalar values in the current attribute set.

        Alternatively, one can choose from a number of preset color maps by
        passing the corresponding string instead.

        =================  =====================================
        Preset color maps
        =================  =====================================
        'jet' (default)    .. image:: images/colormap_jet.png
        'hsv'              .. image:: images/colormap_hsv.png
        'hot'              .. image:: images/colormap_hot.png
        'cool'             .. image:: images/colormap_cool.png
        'spring'           .. image:: images/colormap_spring.png
        'summer'           .. image:: images/colormap_summer.png
        'autumn'           .. image:: images/colormap_autumn.png
        'winter'           .. image:: images/colormap_winter.png
        'gray'             .. image:: images/colormap_gray.png
        =================  =====================================

        Examples:
            >>> xyz = np.c_[np.arange(10), np.zeros(10), np.zeros(10)]
            >>> scalars = np.arange(10)
            >>> v = pptk.viewer(xyz, scalars)
            >>> v.color_map('cool', scale=[0, 5])
            >>> v.color_map([[0, 0, 0], [1, 1, 1]])

        """
        # accepts array of rgb or rgba vectors
        if isinstance(c, str):
            c = _color_maps[c]
        elif isinstance(c, list):
            c = numpy.array(c)
        if len(c.shape) != 2 or c.shape[1] != 3 and c.shape[1] != 4:
            raise ValueError('Expecting array of rgb/rgba vectors')
        if c.shape[1] == 3:
            c = numpy.c_[c, numpy.ones(c.shape[0])]
        self.set(color_map=c)
        if scale is None:
            self.set(color_map_scale=[0, 0])
        else:
            self.set(color_map_scale=scale)

    def capture(self, filename):
        """

        Take screen shot of current view and save to filename

        Examples:

        >>> v = pptk.viewer(xyz)
        >>> v.capture('screenshot.png')

        """
        msg = struct.pack('b', 6) + _pack_string(os.path.abspath(filename))
        self.__send(msg)

    def play(self, poses, ts=[], tlim=[-numpy.inf, numpy.inf], repeat=False,
             interp='cubic_natural'):
        """

        Plays back camera path animation specified by poses

        Args:
            poses: Key poses. e.g. a list of 6-tuples (x, y, z, phi, theta, r)
                poses, or anything convertible to a 6-column array by np.array
            ts (optional): Key pose times. If unspecified key poses are placed
                at 1 second intervals.
            tlim (optional): Play back time range (in seconds)
            repeat (optional): Toggles infinite play back loop.  Works well
                with interp='cubic_periodic'.
            interp (optional): Interpolation method.  Should be one of
                'constant', 'linear', 'cubic_natural', or 'cubic_periodic'.

        Examples:

        Rotate camera about origin at 1/8 Hz.

        >>> poses = []
        >>> poses.append([0, 0, 0, 0 * np.pi/2, np.pi/4, 5])
        >>> poses.append([0, 0, 0, 1 * np.pi/2, np.pi/4, 5])
        >>> poses.append([0, 0, 0, 2 * np.pi/2, np.pi/4, 5])
        >>> poses.append([0, 0, 0, 3 * np.pi/2, np.pi/4, 5])
        >>> poses.append([0, 0, 0, 4 * np.pi/2, np.pi/4, 5])
        >>> v.play(poses, 2 * np.arange(5), repeat=True, interp='linear')

        """
        poses, ts = _fix_poses_ts_input(poses, ts)
        if poses.size == 0:
            return
        msg = struct.pack('b', 8) \
            + struct.pack('i', poses.shape[0]) + poses.tostring() \
            + struct.pack('i', ts.size) + ts.tostring() \
            + struct.pack('b', _interp_code[interp])
        self.__send(msg)
        msg = struct.pack('b', 9) \
            + struct.pack('2f', *tlim) \
            + struct.pack('?', repeat)
        self.__send(msg)

    def record(self, folder, poses, ts=[], tlim=[-numpy.inf, numpy.inf],
               interp='cubic_natural', shutter_speed=numpy.inf, fps=24,
               prefix='frame_', ext='png'):
        """

        Records camera animation to a sequence of images. Usage of this method
        is very similar to viewer.play(...).

        Args:
            folder: Folder to which images are saved
            poses: Same as in :meth:`pptk.viewer.play`
            ts: Same as in :meth:`pptk.viewer.play`
            tlim: Same as in :meth:`pptk.viewer.play`
            interp: Same as in :meth:`pptk.viewer.play`
            fps: Frames per second
            prefix: Resulting image file names are prefixed with this string
            ext: Image format

        Examples:

            Assuming poses defined as in the example for :meth:pptk.viewer.play

            >>> mkdir 'recording'
            >>> v.record('recording', poses)

        Tip: Uses ffmpeg to generate a video from the resulting image sequence

            >>> ffmpeg -i "frame_%03d.png" -c:v mpeg4 -qscale:v 0 -r 24 output.mp4

        """
        if not os.path.isdir(folder):
            raise ValueError('invalid folder provided')
        poses, ts = _fix_poses_ts_input(poses, ts)
        if poses.size == 0:
            return
        # load camera path
        msg = struct.pack('b', 8) + \
            struct.pack('i', poses.shape[0])+poses.tostring() + \
            struct.pack('i', ts.size)+ts.tostring() + \
            struct.pack('b', _interp_code[interp])
        self.__send(msg)

        # clamp tlim[0] and tlim[1] to [ts[0],ts[-1]]
        t_beg = numpy.minimum(numpy.maximum(ts[0], tlim[0]), ts[-1])
        t_end = numpy.minimum(numpy.maximum(ts[0], tlim[1]), ts[-1])

        # ensure t_beg <= t_end
        t_end = numpy.maximum(t_end, t_beg)

        # pose and capture
        num_frames = 1 + numpy.floor((t_end - t_beg) * fps)
        num_digits = 1 + numpy.floor(numpy.log10(num_frames))
        for i in range(int(num_frames)):
            t = i * 1.0 / fps + t_beg
            msg = struct.pack('b', 9) + \
                struct.pack('2f', t, t) + \
                struct.pack('?', False)
            self.__send(msg)
            filename = prefix \
                + ('%0' + str(num_digits) + 'd') % (i + 1) + '.' + ext
            filename = os.path.join(folder, filename)
            self.capture(filename)
            # todo: need to check whether write succeeded
            #       ideally, capture(...) should return filename

    def wait(self):
        """

        Blocks until Enter/Return key is pressed in viewer

        Examples:

            >>> v = pptk.viewer(xyz)
            >>> v.wait()

        Press enter in viewer to return control to python terminal.

        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('localhost', self._portNumber))
        s.send(struct.pack('b', 7))
        s.setblocking(1)
        buf = b''
        while len(buf) == 0:
            buf += s.recv(1)
        if buf != b'x':
            raise RuntimeError('expecting return code \'x\'')
        s.close()

    def __load(self, positions):
        # if no points, then done
        if positions.size == 0:
            return
        # construct message
        numPoints = int(positions.size / 3)
        msg = struct.pack('b', 1) \
            + struct.pack('i', numPoints) + positions.tostring()
        # send message to viewer
        self.__send(msg)

    def __send(self, msg):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('localhost', self._portNumber))
        totalSent = 0
        while totalSent < len(msg):
            sent = s.send(msg)
            if sent == 0:
                raise RuntimeError("socket connection broken")
            totalSent = totalSent + sent
        s.close()

    def __query(self, msg):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # s.setsockopt(socket.SOL_SOCKET,socket.TCP_NODELAY,1)
        s.connect(('localhost', self._portNumber))
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
        lookupSize = {0: 1, 1: 1, 2: 4, 3: 4, 4: 4}
        dataType = ord(s.recv(1))
        numDims = struct.unpack('Q', _recv_from_socket(8, s))[0]
        dims = struct.unpack(str(numDims) + 'Q',
                             _recv_from_socket(numDims * 8, s))
        numElts = numpy.prod(dims)
        bodySize = lookupSize[dataType] * numElts
        body = _recv_from_socket(bodySize, s)
        s.close()

        if dataType == 0:
            raise ValueError(body)
        if dataType != 0 and dataType != 1:
            lookupCode = {1: 'c', 2: 'f', 3: 'i', 4: 'I'}
            body = struct.unpack(str(numElts) + lookupCode[dataType], body)
            body = numpy.array(list(body)).reshape(dims)
        # return body as is if type is char (0)
        return body


def _recv_from_socket(n, s):
    # receive n bytes from socket s
    buf = b''
    while len(buf) < n:
        buf += s.recv(n - len(buf))
    return buf


def _fix_poses_ts_input(poses, ts):
    # ensure poses is 6 column array of floats
    poses = numpy.float32(numpy.array(poses).reshape(-1, 6)).copy()

    # ensure ts has the same number of timestamps as poses
    ts = numpy.float32(numpy.array(ts))
    if ts.size == 0:
        ts = numpy.float32(numpy.arange(poses.shape[0]))
    elif ts.size != poses.shape[0]:
        raise ValueError('number of time stamps != number of key poses')

    # ensure ts is unique and ascending
    if numpy.any(numpy.diff(ts) <= 0):
        raise ValueError('time stamps must be unique and ascending')

    # ensure subsequent angle differences between -180 and +180 degrees
    def correct_angles(x):
        # note: mapping takes +180 + 360k to +180,
        #       and -180 + 360k to -180, for any integer k
        d = numpy.diff(x)
        absd = numpy.abs(d)
        y = -absd - 2.0 * numpy.pi \
                        * numpy.floor((-absd + numpy.pi) / 2.0 / numpy.pi)
        y *= -numpy.sign(d)
        return x[0] + numpy.r_[0, numpy.cumsum(y)]
    poses[:, 3] = correct_angles(poses[:, 3])
    poses[:, 4] = correct_angles(poses[:, 4])

    return (poses, ts)


def _encode_bool(x):
    try:
        y = struct.pack('?', x)
    except Exception:
        raise
    return y


def _encode_float(x):
    try:
        y = struct.pack('f', x)
    except Exception:
        raise
    return y


def _encode_floats(x):
    return numpy.asarray(x, dtype=numpy.float32).tostring()


def _encode_uints(x):
    return numpy.asarray(numpy.uint32(x)).tostring()


def _encode_uint(x):
    try:
        y = struct.pack('I', x)
    except Exception:
        raise
    return y


def _encode_rgb(x):
    x = numpy.asarray(numpy.float32(x))
    if x.size != 3 or numpy.any(numpy.logical_or(x < 0.0, x > 1.0)):
        raise ValueError('Expecting 3 values in [0,1]')
    return struct.pack('fff', x[0], x[1], x[2])


def _encode_rgba(x):
    x = numpy.asarray(numpy.float32(x))
    if x.size != 4 or numpy.any(numpy.logical_or(x < 0.0, x > 1.0)):
        raise ValueError('Expecting 4 values in [0,1]')
    return struct.pack('ffff', x[0], x[1], x[2], x[3])


def _encode_rgbas(x):
    x = numpy.asarray(numpy.float32(x))
    if x.shape[1] == 4 and numpy.all(numpy.logical_and(x >= 0.0, x <= 1.0)):
        return x.tostring()
    else:
        raise ValueError('Expecting 4 column array of values in [0,1]')


def _encode_xyz(x):
    x = numpy.asarray(numpy.float32(x))
    if x.size != 3:
        raise ValueError('Expecting 3 values')
    return struct.pack('fff', x[0], x[1], x[2])


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
    return struct.pack('b', 5) + _pack_string(prop_name)


def _construct_set_msg(prop_name, prop_value):
    if not _properties.get(prop_name):
        raise ValueError('Invalid property name encountered: %s' % prop_name)
    msg_header = struct.pack('b', 4) + _pack_string(prop_name)
    msg_payload = ''
    try:
        msg_payload = _properties[prop_name](prop_value)
    except BaseException as e:
        raise ValueError('Failed setting "%s": ' % prop_name + str(e))
    return msg_header + struct.pack('Q', len(msg_payload)) + msg_payload


def _pack_string(string):
    return struct.pack('Q', len(string)) + \
        struct.pack(str(len(string)) + 's', string.encode('ascii'))


def _init_color_maps():
    _color_maps['jet'] = numpy.array(
        [[0, 0, 1],
         [0, 1, 1],
         [0, 1, 0],
         [1, 1, 0],
         [1, 0, 0]], dtype=numpy.float32)
    _color_maps['hsv'] = numpy.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1],
         [1, 0, 0]], dtype=numpy.float32)
    _color_maps['hot'] = numpy.array(
        [[0, 0, 0],
         [1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]], dtype=numpy.float32)
    _color_maps['cool'] = numpy.array(
        [[0, 1, 1],
         [1, 0, 1]], dtype=numpy.float32)
    _color_maps['spring'] = numpy.array(
        [[1, 0, 1],
         [1, 1, 0]], dtype=numpy.float32)
    _color_maps['summer'] = numpy.array(
        [[0, .5, .4],
         [1, 1, .4]], dtype=numpy.float32)
    _color_maps['autumn'] = numpy.array(
        [[1, 0, 0],
         [1, 1, 0]], dtype=numpy.float32)
    _color_maps['winter'] = numpy.array(
        [[0, 0, 1],
         [0, 1, .5]], dtype=numpy.float32)
    _color_maps['gray'] = numpy.array(
        [[0, 0, 0],
         [1, 1, 1]], dtype=numpy.float32)


_properties = dict()
_init_properties()
_color_maps = dict()
_init_color_maps()

# define codes for each interpolation scheme
_interp_code = {'constant': 0,
                'linear': 1,
                'cubic_natural': 2,
                'cubic_periodic': 3}
