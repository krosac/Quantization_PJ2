# Quantization_tvm_tf
## Build
Follow https://docs.tvm.ai/install/from_source.html to install prerequisites. Anaconda or virtualenv is recommended. 
```
sudo apt-get update
sudo apt-get install -y python python-dev python-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake
sudo apt-get install -y ocl-icd-opencl-dev libxml2-dev
sudo pip3 install decorator antlr4-python3-runtime attrs scipy
```
opencv
```
sudo apt-get install libsm6 libxrender1 libfontconfig1
sudo apt-get install opencv-python
```
Also install Tensorflow(<2.0) for .pb file or PyTorch for .onnx. 
```
sudo pip install tensorflow
sudo pip install torch torchvision
```
Then we install TVM stack.
```
git clone --recursive https://github.com/apache/incubator-tvm tvm
```
Set environment variables. (You can update ~/.bashrc so that every time you log on they will be automatically exported.)
```
export TVM_HOME=/your/tvm/root/path
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python
```
Get our Quantization_PJ2 folder. Run script inside to add files in TVM folder. **Make sure pre-compiled .so files match your python version and platform**. If not, try to generate .so files for **each** .c file following commands below.
```
gcc -pthread -B /my/conda/env/path/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/my/conda/env/path/include/your-python3.xm -c XXX.c -o XXX.o
gcc -pthread -shared -B /my/conda/env/path/compiler_compat -L/my/conda/env/path/lib -Wl,-rpath=/my/conda/env/path/lib -Wl,--no-as-needed -Wl,--sysroot=/ XXX.o -o XXX.so
```
Run script.
```
mv /folder/match/your/platform utils
sh patch.sh
```
Prepare cmake configurations.
```
cd $TVM_HOME
cp cmake/config.cmake .
```
Config LLVM path in config.cmake. First download pre-built llvm binary at http://releases.llvm.org/download.html#9.0.0 . Here we take 9.0.0 for Ubuntu 16.04 as the example.
```
wget http://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
tar xvf clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
```
Edit config.cmake to specify **absolute path** for llvm-config: change line 121 to set(USE_LLVM path/to/bin/llvm-config). Following the example above, we have
```
set(USE_LLVM /absolute/path/to/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-16.04/bin/llvm-config)
```
Then compile the code. This could take a while.
```
mkdir build
cd build
cmake ..
make -j4
```
If it completes to 100% without error, we build TVM successfully. 
## Implement quantization functions
**You are required to complete $TVM_HOME/python/tvm/relay/utils/quantization_utils.py**

The following command quantizes the input model and dumps quantized weights to output_dir.
```
python from_tensorflow.py --input ~/tvm/example/tests/shufflenet_v1/ShuffleNet-1g8-Pytorch/shufflenet_v1.onnx --input_shape 1 224 224 3 --output_dir tmp --gen_pb --preprocess torchvision --gen_fx_pb --reference_input ILSVRC2012_val_00001110.JPEG --dump
```



