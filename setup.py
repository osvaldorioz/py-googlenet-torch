from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'googlenet',
        [
            'main.cpp',
            'domain/GoogLeNetService.tpp'  # Todavía usamos este .tpp
        ],
        include_dirs=[
            pybind11.get_include(),
            '/home/hadoop/libtorch/include',
            '/home/hadoop/libtorch/include/torch/csrc/api/include',
            'domain',
            'ports',
            'adapters'
        ],
        library_dirs=['/home/hadoop/libtorch/lib'],
        libraries=['torch', 'torch_cpu', 'c10'],
        language='c++',
        extra_compile_args=['-std=c++17'],
        extra_link_args=['-Wl,-rpath,/home/hadoop/libtorch/lib']
    ),
]

setup(
    name='googlenet',
    version='0.1',
    ext_modules=ext_modules,
)