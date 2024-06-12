Grace Workflow
==============
python -m venv ${HOME}/software/venv_pytorch
source ${HOME}/software/venv_pytorch/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

mkdir catch2
wget https://github.com/catchorg/Catch2/releases/download/v2.13.9/catch.hpp -O ./catch2/catch.hpp

git clone https://github.com/libxsmm/libxsmm.git
cd libxsmm
make BLAS=0 -j
cd ..

CXXFLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 make all

LIBXSMM_TARGET=aarch64 ./build/test
LIBXSMM_TARGET=aarch64 ./build/performance_matmul
