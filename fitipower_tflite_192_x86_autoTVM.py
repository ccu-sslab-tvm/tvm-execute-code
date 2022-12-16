'''
使用模型： yolov5_704_p-0.9474_r-0.9408_map50-0.6404_192x192_ch1_ReLU-int8.tflite

在電腦上執行，包含 autoTVM 功能

天鈺模型，已可以正確辨識天鈺的資料集

目前已知問題： 無
'''

import os
import tarfile
from datetime import datetime

import numpy
import tvm
from PIL import Image
from tvm import autotvm, relay, transform
from tvm.autotvm.tuner import XGBTuner
from tvm.contrib import graph_executor
from tvm.relay.backend import Executor

from post_process import post_process_fitipower as post

# path setting
output_folder_path = './test_outputs'
output_path = output_folder_path + '/fitipower_tflite_192_x86_autoTVM'
model_folder_path = './model'
img_folder_path = './img/'
tvm_temp_path = '/home/yang880519/tvm_temp' # Warning：This folder will be removed every time.

# TVM IR output setting
IR_output = True # Output relay & params or not
transfer_layout = True # 是否進行 layout 轉換
original_relay_path = output_path + '/original_realy.txt'
original_params_path = output_path + '/original_params.txt'
converted_relay = output_path + '/converted_mod.txt'

# model using setting
model_path = model_folder_path + '/yolov5_704_p-0.9474_r-0.9408_map50-0.6404_192x192_ch1_ReLU-int8.tflite'
input_name = 'input_1_int8'
input_shape = (1, 192, 192, 1)
input_dtype = 'int8'

# input image setting
img_name = '202205051553337.jpeg'
img_path = img_folder_path + img_name

# optimize setting
opt_level = 3

# autoTVM setting
use_autoTVM = True
retune = True
number = 100
repeat = 50
trials = 1500
early_stopping = 100
min_repeat_ms = 0 # since we're tuning on a CPU, can be set to 0
timeout = 100
records_path = output_path + '/autoTVM.json'

# make C code
output_c_code = True
tar_file_path = output_path + '/c_code.tar'

# executor mode
executor_mode = 'graph' # 'graph' or 'aot'

#------------------------------------------------------------------------------
# make output folder
if not os.path.exists(output_path):
    os.mkdir(output_path)

# load model by the frontend
model_buffer = open(model_path, 'rb').read()

try:
    import tflite
    model = tflite.Model.GetRootAsModel(model_buffer, 0)
except AttributeError:
    import tflite.Model
    model = tflite.Model.Model.GetRootAsModel(model_buffer, 0)

# preprocesss input image
img_data = Image.open(img_path).resize((192, 192))
img_data = img_data.convert('L')
img_data = numpy.array(img_data) - 128 # 量化到 int8 空間
img_data = numpy.expand_dims(img_data, axis = 0)
img_data = numpy.expand_dims(img_data, axis = -1)
img_data = numpy.asarray(img_data).astype('int8')

# load frontend model to TVM IR
mod, params = relay.frontend.from_tflite(
    model = model,
    shape_dict = {input_name: input_shape},
    dtype_dict = {input_name: input_dtype}
)

if IR_output:
    print(mod, file = open(original_relay_path, 'w'))
    print(params, file = open(original_params_path, 'w'))

if transfer_layout:
    desired_layouts = {'qnn.conv2d': ['NCHW', 'default'], 'nn.max_pool2d':['NCHW', 'default'], 'image.resize2d':['NCHW']}
    seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)]) #relay.transform.RemoveUnusedFunctions()
    with tvm.transform.PassContext(opt_level = opt_level):
        mod = seq(mod)

if IR_output:
    print(mod, file = open(converted_relay, 'w'))

# set target information
TARGET = 'llvm'
EXECUTOR = Executor('graph') if executor_mode == 'graph' else Executor("aot")

#autoTVM tuning
if use_autoTVM:
    builder = autotvm.LocalBuilder()
    runner = autotvm.LocalRunner(
        number = number,
        repeat = repeat,
        timeout = timeout,
        min_repeat_ms = min_repeat_ms,
        enable_cpu_cache_flush = True,
    )
    measure_option = autotvm.measure_option(
        builder = builder, 
        runner = runner
    )
    tuning_option = {
        'trials': trials,
        'early_stopping': early_stopping,
        'measure_option': measure_option,
        'tuning_records': records_path,
    }
    if retune:
        if os.path.exists(tuning_option['tuning_records']):
            os.remove(tuning_option['tuning_records'])

        tasks = autotvm.task.extract_from_program(mod['main'], params = params, target = TARGET)
        assert len(tasks) > 0

        for i, task in enumerate(tasks):
            prefix = '[%s][Task: %2d/%2d] ' % (str(datetime.now().strftime("%Y/%m/%d %H:%M:%S")), i + 1, len(tasks))
            tuner_obj = XGBTuner(task, loss_type = 'rank')
            tuner_obj.tune(
                n_trial = min(tuning_option['trials'], len(task.config_space)),
                early_stopping = tuning_option['early_stopping'],
                measure_option = tuning_option['measure_option'],
                callbacks = [
                    tvm.autotvm.callback.progress_bar(tuning_option['trials'], prefix = prefix),
                    tvm.autotvm.callback.log_to_file(tuning_option['tuning_records']),
                ],
            )

# compile relay to tir
if use_autoTVM:
    with autotvm.apply_history_best(tuning_option['tuning_records']):
        with transform.PassContext(
            opt_level = opt_level, 
            config = {'tir.disable_vectorize': True}, 
            disabled_pass = ['AlterOpLayout']
        ):
            lib = relay.build(mod, target = TARGET, executor = EXECUTOR, params = params)
else:
    with transform.PassContext(
        opt_level = opt_level, 
        config = {'tir.disable_vectorize': True}, 
        disabled_pass = ['AlterOpLayout']
    ):
        lib = relay.build(mod, target = TARGET, executor = EXECUTOR, params = params)

# make C code file
if output_c_code:
    tvm.micro.export_model_library_format(lib, tar_file_path)
    with tarfile.open(tar_file_path, 'r:*') as tar_f:
        print('\n'.join(f' - {m.name}' for m in tar_f.getmembers()))

#run and time testing-----------------------------------------------------------
dev = tvm.device(TARGET, 0)

if executor_mode == 'graph':
    module = graph_executor.GraphModule(lib['default'](dev))
elif  executor_mode == 'aot':
    temp_dir = tvm.contrib.utils.tempdir(tvm_temp_path)
    test_so_path = temp_dir / "test.so"
    lib.export_library(test_so_path, cc="gcc", options=["-std=c11", "-g3", "-O0"])
    loaded_mod = tvm.runtime.load_module(test_so_path)
    module = tvm.runtime.executor.AotModule(loaded_mod['default'](dev))

dtype = 'int8'
module.set_input(input_name, img_data)

time_start = datetime.now()
module.run()
time_end = datetime.now() # 計算 graph_mod 的執行時間
print('spent {0}', time_end - time_start)
if executor_mode == 'graph': # AoT Mode 不支援 benchmark
    print('------------------------------TVM benchmark------------------------------')
    print(module.benchmark(dev, number = 100, repeat = 3))

tvm_output = module.get_output(0).numpy()

#post process--------------------------------------------------------------------
post.post_process(tvm_output[0], output_path, img_path, img_name)