import subprocess
import os
import ctypes

def eigen():
    return ['-I/usr/local/include/eigen3']


def sophus():
    return ['-I/Users/adela/projects/Sophus']


def json():
    return ['-I/Users/adela/projects/json/single_include/nlohmann']


def opencv():
    libs = subprocess.Popen(['pkg-config', '--libs', 'opencv'], stdout=subprocess.PIPE)
    includes = subprocess.Popen(['pkg-config', '--cflags', 'opencv'], stdout=subprocess.PIPE)
    flags = []
    for line in libs.stdout:
        flags.append(line.rstrip())
    for line in includes.stdout:
        flags.append(line.rstrip())
    return flags


def ceres():
    return ['-I/usr/local/include', '/usr/local/lib/libceres.a', '-lceres']


def glog():
    return ['/usr/local/lib/libglog.dylib']


def gflags():
    return ['/usr/local/lib/libgflags.2.2.0.dylib']


def cflags():
    return ['-O3', '-ffast-math', '-funroll-loops', '-fPIC']


def compile(directory, source):
    output_filenaem = os.path.join(directory, source + '.so')
    source_filename = os.path.join(directory, source + '.cc')
    print ("Compiling", os.path.join(directory, source + '.cc'))
    command = ["clang++", "-shared", "-std=c++17", "-o", output_filenaem, source_filename]
    subprocess.call(command + cflags() + eigen() + ceres() + opencv() + gflags() + glog() + sophus())


def cfunc(name, dll, result, *args):
    '''build and apply a ctypes prototype complete with parameter flags'''
    atypes = []
    aflags = []
    for arg in args:
        atypes.append(arg[1])
        aflags.append((arg[2], arg[0]) + arg[3:])
    return ctypes.CFUNCTYPE(result, *atypes)((name, dll), tuple(aflags))


class c_array(object):
    '''Just like a POINTER but accept a list of ctype as an argument'''
    def __init__(self, etype):
        self.etype = etype

    def from_param(self, param):
        if isinstance(param, (list, tuple)):
            return (self.etype * len(param))(*param)


class c_matrix(object):
    '''Just like POINTER(POINTER(ctype)) but accept a list of lists of ctype'''
    def __init__(self, etype):
        self.etype = etype

    def from_param(self, param):
        if isinstance(param, (list, tuple)):
            val = (ctypes.POINTER(self.etype) * len(param))()
            for i,v in enumerate(param):
                if isinstance(v, (list, tuple)):
                    val[i] = (self.etype * len(v))(*v)
                else:
                    raise TypeError('nested list or tuple required at %d' % i)
            return val
        else:
            raise TypeError('list or tuple required')


class c_ref(object):
    '''Just like a POINTER but accept an argument and pass it byref'''
    def __init__(self, atype):
        self.atype = atype

    def from_param(self, param):
        return ctypes.byref(self.atype(param))


def matrix_to_ctype_array(src):
    row = len(src)
    col = len(src[0])
    print(row, col)
    size = row * col;
    pose_array = ctypes.c_double * size
    dst = pose_array()

    for i in range(row):
        for j in range(col):
            dst[i*col + j] = ctypes.c_double(src[i][j]).value
    return dst