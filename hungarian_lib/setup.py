from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='hungarian_lib',
      ext_modules=[cpp_extension.CppExtension('hungarian_lib', ['hungarian_lib.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

'''
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

LIBTORCH='libtorch'

setup(
	name='hungarian_lib', 
	ext_modules = [CppExtension(name = 'hungarian_lib', sources = ['hungarian_lib.cpp'])], 
	library_dirs = [
		LIBTORCH + '/include', 
		LIBTORCH + '/include/torch/csrc/api/include',
		LIBTORCH + '/include/TH',
		LIBTORCH + '/include/THC'],
	cmdclass = {'build_ext': BuildExtension})
'''
