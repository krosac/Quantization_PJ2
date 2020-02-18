# Quantization Tool for 209 Project 2
Thanks for taking the class. It could take time if you are not familiar with buidling software frameworks. Don't worry and follow the steps below. Hope you can get started soon:) Feel free to send emails to me if there is any question. Check project 2 slides on ccle for email address. (Installation has been verified on AWS with Ubuntu16.04. It should also work on your own workstation.)

**Note:** The guide below is for Linux system. If you prefer Windows system, a pre-built version is also provided as utils_py37_x86_win64. In that case you would need to install TVM (and possibly LLVM) on Windows from scratch by replacing build_module.cc at tvm/src/relay/backend first. Then copy utils folder to tvm/python/tvm/relay/. We recommend you to work on Linux, if possible, because building code on Windows could be tricky and lack of documentations.

## Build
Install prerequisites first. If still dependecies are missing, follow https://docs.tvm.ai/install/from_source.html for more details. Anaconda or virtualenv is recommended. 
```
sudo apt-get update
sudo apt-get install -y python python-dev python-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake
sudo apt-get install -y ocl-icd-opencl-dev libxml2-dev
sudo pip3 install decorator antlr4-python3-runtime attrs scipy ipdb
```
opencv
```
sudo apt-get install libsm6 libxrender1 libfontconfig1
sudo apt-get install python-opencv
```
Also install Tensorflow(<2.0) for .pb file or PyTorch for .onnx. This is the cpu version. (If you would like to install gpu version, please check out on their offical websites.)
```
sudo pip install tensorflow
sudo pip install torch torchvision
```
Then we install TVM stack. Here we will use stable old version due to the frequent updates in TVM official repository. Latest version could be found at https://github.com/apache/incubator-tvm.
```
git clone --recursive https://github.com/krosac/incubator-tvm.git tvm
```
Set environment variables. (You can update ~/.bashrc so that every time you log on they will be automatically exported.)
```
export TVM_HOME=/your/tvm/root/path
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python
```
Get our Quantization_PJ2 folder. Run script patch.sh to add files in TVM folder. **Make sure pre-compiled .so files match your python version and platform**. We provide pre-built version on x86 platform with python3.5/3.7. If not, try to generate .so files for **each** .c file following commands below. **(Skip this step if version matches)**
```
gcc -pthread -B /my/conda/env/path/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/my/conda/env/path/include/your-python3.xm -c XXX.c -o XXX.o
gcc -pthread -shared -B /my/conda/env/path/compiler_compat -L/my/conda/env/path/lib -Wl,-rpath=/my/conda/env/path/lib -Wl,--no-as-needed -Wl,--sysroot=/ XXX.o -o XXX.so
```
If version matches, modify folder name and run script.
```
mv utils_pyxx_xxx utils # modify the folder name here i.e. mv utils_py35_x86 utils
sh patch.sh # cp utils to tvm/relay
```
Then we continue to build TVM. Prepare cmake configurations.
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

Try the command below to generate full-precision Tensorflow .pb
```
python3 main.py --input shufflenet_v1.onnx --input_shape 1 224 224 3 --output_dir tmp --gen_pb
```
The following command quantizes the input model and dumps quantized weights to output_dir.
```
python3 main.py --input shufflenet_v1.onnx --input_shape 1 224 224 3 --output_dir tmp --gen_pb --preprocess torchvision --gen_fx_pb --reference_input ILSVRC2012_val_00001110.JPEG --dump
```
ILSVRC2012_val_00001110.JPEG is a reference input sampled from ImageNet ILSVRC2012 validation set. Please check preprossing method for your model, which is important to get correct evaluation result.

## Model List
Model | Input Shape | Source
------|--------|--------|
resnet v1 50|	224x224x3	|https://pytorch.org/docs/stable/torchvision/models.html
resnet v2 50|	224x224x3	|https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py
yolov3|	416x416x3	|https://github.com/eriklindernoren/PyTorch-YOLOv3
mobilenet v1	|224x224x3	|https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md
mobilenet v2	|224x224x3	|https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
mobilenet v3	|224x224x3	|https://github.com/xiaolai-sqlai/mobilenetv3/blob/master/mobilenetv3.py
densenet161	|224x224x3	|https://pytorch.org/docs/stable/torchvision/models.html
shufflenet_v1	|224x224x3	|https://github.com/ericsun99/ShuffleNet-1g8-Pytorch
xception	|229x229x3	|https://github.com/tstandley/Xception-PyTorch
squeezenet	|224x224x3	|https://pytorch.org/docs/stable/torchvision/models.html

You can use basic 1.0 224 configuration for MobileNets

## Evaluation
If ImageNet official site doesn't work, try http://academictorrents.com/browse.php?search=imagenet with ```transmission-cli``` on Linux. Ground truth labels are also uploaded here as val.txt.

Also, I have uploaded ImageNet 2012 validation set to https://drive.google.com/file/d/1_yXhr4wXlyJd2EOB69i7CGikJFWG5hiJ/view?usp=sharing in case provided links cannot work.

For YOLOv3 evaluation, please check its source link above for more details. You can download COCO dataset with the scripts provided in that link. Evaluation framework is also provided there. Just make some changes for your .pb file.

## Debugging
Input and output tensor names in generated test.pb are "input_fx" and "output_fx". YOLOv3 has mutiple output tensors. Specify them with "--output_tensor_names output_fx BiasAdd_58 BiasAdd_66". Replace tensor names with names in your model. 

By adding "--dump" flag, main.py generates weights/biases/intermediate feature maps to the directory specified by "--output_dir". Compare quantized ones with full-precison ones to make sure your implementation works correctly.

Example terminal output with quantization-related flags applied: 
![alt text](https://github.com/krosac/Quantization_PJ2/blob/master/quantization_screenshot.JPG)

"fm_9", "weight_9", "bias_9" would be found in the directory specified by "--output_dir". "9" indicates the layer index. Let's take a look line by line.

"<<< feature map 1/8 >>>" incdicates the fraction length, which is the output of your search(), is 1 and word length is 8. After quantization of intermediate feature map, conv2d is performed with weight quantized to fraction length=8 and word lenght=8. "<1>" represents the number of groups for the conv2d above. If it is 1, then it is a converntional conv2d. If it equals the input channel number, then it is a depthwise conv2d. For example, if we have "<128>" here, it is a depthwise conv2d. Then "(1,27,27,128)" stands for the output shape of conv2d, where four dimensions are NHWC. After that, bias addition is applied. Bias is quantized to fraction length=18, word length=16. Finally, relu is applied to the output of bias addition. Output shapes of bias addtion and relu are also listed below the operation names.

