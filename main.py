from tvm import relay
import argparse
import os
######################################################################
# Command line arguments
parser = argparse.ArgumentParser(description='TVM-based frontend for FPGA deployment')
parser.add_argument('--input',metavar='NAME', type=str, nargs='?', default='', help='file name(.pb/.onnx)')
parser.add_argument('--input_shape',metavar='SHAPE', type=int, nargs='+', default='', help='i.e. 1 224 224 3')
parser.add_argument('--output_dir',metavar='DIR', type=str, nargs='?', default='./output/', help='output dir for generated weights')
parser.add_argument('--gen_pb', type=bool, nargs='?', const=True, default=False, help='generate pb')
parser.add_argument('--gen_fx_pb', type=bool, nargs='?', const=True, default=False, help='generate pb for OPU simulation')
parser.add_argument('--merge_bn', type=bool, nargs='?', const=True, default=False, help='merge bn mul&add to conv+bias as possible') # not work for now
parser.add_argument('--preprocess', type=str, nargs='?', default='', help='preprocess method used for evaluation')
parser.add_argument('--dump', type=bool, nargs='?', const=True, default=False, help='dump weights&ofm to <output_dir>')
parser.add_argument('--darknet_darkflow', type=bool, nargs='?', const=True, default=False, help='indicate whether input pb file is converted from darknet by darkflow') # not work for now
parser.add_argument('--reference_input', type=str, nargs='?', default='', help='Input image for fraction length')
parser.add_argument('--output_tensor_names',type=str, nargs='+', default=['output_fx'], help='specify output tensor names if there are more than one')
parser.add_argument('--dump_format',type=str, nargs='?', default='npy', help='specify output weights format i.e. npy/mat')
args = parser.parse_args() 

mpath = args.input
input_shape = tuple(args.input_shape)
fx_output_dir = args.output_dir
config_dict = {
    'gen_pb':args.gen_pb,\
    'gen_fx_pb':args.gen_fx_pb,\
    'merge_bn':args.merge_bn,\
    'preprocess':args.preprocess,\
    'dump':args.dump,\
    'input_shape':input_shape,\
    'output_dir':fx_output_dir,\
    'ref_input':args.reference_input,\
    'out_names':args.output_tensor_names,\
    'dump_format':args.dump_format
    }
if args.gen_fx_pb:
    assert not args.reference_input=='','reference_input needs to be specified for quantization'
    assert not args.preprocess=='','preprocessing method needs to be speceified for quantization'
if args.preprocess=='custom':
    assert args.reference_input.endswith(('.npy')),'custom reference_input is required to be preprocessed .npy file'
if args.dump and os.path.exists(args.output_dir):
    files = os.listdir(args.output_dir)
    if any(file.endswith(('.mat')) for file in files):
        os.system('rm '+os.path.join(args.output_dir,'*.mat'))
    if any(file.endswith(('.npy')) for file in files):
        os.system('rm '+os.path.join(args.output_dir,'*.npy'))
if args.dump:
    assert args.dump_format in ['npy','mat'],'dump_format needs to be mat/npy'
######################################################################
# Import the graph to Relay
# -------------------------
# Import tensorflow/pytorch graph definition to relay frontend.
#
# Results:
#   sym: relay expr for given tensorflow protobuf/pytorch onnx.
#   params: params converted from tensorflow/pytorch params.
if mpath.endswith(('.onnx')):
    import onnx
    onnx_ishape = [input_shape[0],input_shape[-1]]+list(input_shape[1:-1])  
    shape_dict = {'input': tuple(onnx_ishape)}
    model = onnx.load(mpath)
    sym, params = relay.frontend.from_onnx(model, shape=shape_dict)
    print ("PyTorch onnx imported to relay frontend.")
    config_dict['platform'] = 'pytorch' #NCHW
elif mpath.endswith(('.h5')):
    import keras
    keras_ishape = [input_shape[0],input_shape[-1]]+list(input_shape[1:-1])  
    shape_dict = {'input_1': tuple(keras_ishape)}
    model = keras.models.load_model(mpath)
    sym, params = relay.frontend.from_keras(model, shape=shape_dict)
    print ("Keras h5 imported to relay frontend.")
    config_dict['platform'] = 'pytorch' #NCHW
elif mpath.endswith(('.pb')):
    import tensorflow as tf
    shape_dict = {'input': input_shape}
    with tf.gfile.FastGFile(mpath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())        
    sym, params = relay.frontend.from_tensorflow(graph_def, layout='NHWC', shape=shape_dict)
    print ("Tensorflow protobuf imported to relay frontend.")
else:
    print('[ERROR] Input model file format not supported (Tensorflow Protobuf & PyTorch ONNX)')
    exit()

from tvm.relay.utils import Quantizer
Q = Quantizer.Quantizer(sym, params, config_dict)
