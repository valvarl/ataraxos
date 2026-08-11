"""Build script for the CUDA C++ extension (_stratego_cuda).

Implements the StrategoRolloutBuffer as a pybind11 class_ holding GPU-resident
torch::Tensor members, with CUDA kernels operating on raw data_ptr<T>() pointers.

Compiles for SM 7.5 (T4), SM 8.6 (RTX 3060), and SM 9.0 (H100) so the same
wheel runs on every target GPU.
"""

from pathlib import Path

import torch
from setuptools import setup

# Conditionally import torch CUDA extension utilities — torch is a build dep
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent
CSRC = ROOT / "csrc"
TORCH_LIB = str(Path(torch.__file__).parent / "lib")

# Sources — only build the CUDA extension when the .cu file exists AND CUDA is available.
cuda_sources = ["csrc/stratego_buffer.cu", "csrc/stratego_buffer.cpp"]
has_cuda = all((CSRC / name).exists() for name in ("stratego_buffer.cu", "stratego_buffer.cpp"))

# nvcc flags: build PTX/SASS for T4 (sm_75), RTX 3060 (sm_86), H100 (sm_90)
nvcc_flags = [
    "-std=c++17",
    "-O3",
    "--use_fast_math",
    "-gencode=arch=compute_75,code=sm_75",  # T4 (sm_70 dropped in CUDA 13)
    "-gencode=arch=compute_86,code=sm_86",  # RTX 3060
    "-gencode=arch=compute_90,code=sm_90",  # H100
    "-gencode=arch=compute_90,code=compute_90",  # PTX forward-compat for future SMs
]
cxx_flags = ["-std=c++17", "-O3"]

if has_cuda:
    ext_modules = [
        CUDAExtension(
            name="_stratego_cuda",
            sources=cuda_sources,
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
            extra_link_args=[f"-Wl,-rpath,{TORCH_LIB}"],
        ),
    ]
    cmdclass = {"build_ext": BuildExtension}
else:
    # Graceful fallback when CUDA sources not yet written (early dev)
    ext_modules = []
    cmdclass = {}

setup(
    name="ataraxos-cuda",
    package_dir={"": "."},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
