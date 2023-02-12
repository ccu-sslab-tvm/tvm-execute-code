'''
使用模型： yolov5_704_p-0.9474_r-0.9408_map50-0.6404_192x192_ch1_ReLU-int8.tflite

在 stm32f429 上執行，包含 autoTVM 功能

天鈺模型，已可以正確辨識天鈺的資料集

--- problem ---
新增 cmsis-nn 使用選擇，但 zephyr 若要支援 cmsis 則需要編譯 library 並提供路徑，目前還沒處理好
因此 zephyr 無法支援 cmsis 的情況下無法使用 autoTVM 進行 tuning
--- problem ---

目前已知問題： 無
'''

import json
import os
import pathlib
import tarfile
from datetime import datetime

import numpy
import tvm
from PIL import Image
from tvm import autotvm, relay, transform
from tvm.autotvm.tuner import XGBTuner
from tvm.driver.tvmc.composite_target import get_codegen_by_target
from tvm.driver.tvmc.pass_config import parse_configs
from tvm.relay.backend import Executor, Runtime

from post_process import post_process_fitipower as post

# model setting
model_name = 'yolov5_704_p-0.9474_r-0.9408_map50-0.6404_192x192_ch1_ReLU-int8.tflite'
size = 192
input_name = 'input_1_int8'
input_shape = (1, size, size, 1)
input_dtype = 'int8'

# input image setting
img_name = '202205051553337.jpeg'

# TVM IR output setting
IR_output = True # Output relay & params or not
transfer_layout = True # 是否進行 layout 轉換
using_cmsis_nn = True

# board using setting
use_board = True # Use board or the simulator
executor_mode = 'aot' # 'graph' or 'aot'
test_time = 1

# autoTVM setting
use_autoTVM = False
retune = True
number = 5
repeat = 3
trials = 20000
early_stopping = 100
min_repeat_ms = 0 # since we're tuning on a CPU, can be set to 0
timeout = 120

# optimize setting
opt_level = 3

# make C code
output_c_code = True

# Zephyr runtime setting
zephyr_run = True

#------------------------------------------------------------------------------
# path setting
output_path = './test_outputs/fitipower_tflite_stm_autoTVM_' + str(size)

model_path = './model/' + model_name
img_path = './img/' + img_name

original_relay_path = output_path + '/original_realy.txt'
original_params_path = output_path + '/original_params.txt'
converted_relay = output_path + '/converted_mod.txt'
cmsis_nn_relay = output_path + '/cmsis_nn_mod.txt'

records_path = output_path + '/autoTVM.json'

tar_file_path = output_path + '/c_code.tar'

tvm_temp_path = '/home/yang880519/tvm_temp' # Warning：This folder will be removed every time.

# board using setting
simulator = 'qemu_x86'
physical_hw = 'stm32f429i_disc1'

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

if using_cmsis_nn:
    config = parse_configs(None)
    extra_targets = [{'name': 'cmsis-nn', 'opts': {'mcpu': 'cortex-m4'}, 'raw': 'cmsis-nn', 'is_tvm_target': False}]
    for codegen_from_cli in extra_targets:
            codegen = get_codegen_by_target(codegen_from_cli['name'])
            partition_function = codegen['pass_pipeline']

            if codegen['config_key'] is not None:
                config[codegen['config_key']] = codegen_from_cli['opts']
            with tvm.transform.PassContext(config=config):
                mod = partition_function(mod, params, mod_name='default', **codegen_from_cli['opts'])
    print(mod, file = open(cmsis_nn_relay, 'w'))

if transfer_layout:
    desired_layouts = {'qnn.conv2d': ['NCHW', 'default'], 'nn.max_pool2d':['NCHW', 'default'], 'image.resize2d':['NCHW']}
    seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)]) #relay.transform.RemoveUnusedFunctions()
    with tvm.transform.PassContext(opt_level = opt_level):
        mod = seq(mod)

if IR_output:
    print(mod, file = open(converted_relay, 'w'))

# set target information
boards_file = pathlib.Path(tvm.micro.get_microtvm_template_projects('zephyr')) / 'boards.json'
with open(boards_file) as f:
    boards = json.load(f)

BOARD = physical_hw if use_board else simulator
TARGET = tvm.target.target.micro(boards[BOARD]['model'] if use_board else 'host')
RUNTIME = Runtime('crt', {'system-lib': True})
EXECUTOR = Executor('graph') if executor_mode == 'graph' else Executor('aot')

#autoTVM tuning
if use_autoTVM:
    module_loader = tvm.micro.AutoTvmModuleLoader(
        template_project_dir = pathlib.Path(tvm.micro.get_microtvm_template_projects('zephyr' if use_board else 'crt')),
        project_options = {
            'zephyr_board': BOARD,
            'west_cmd': 'west',
            'verbose': False,
            'project_type': 'host_driven'
        } if use_board else {
            'verbose': False
        }
    )
    builder = autotvm.LocalBuilder(
        build_kwargs = {'build_option': {'tir.disable_vectorize': True}},
        do_fork = False if use_board else True,
        build_func = tvm.micro.autotvm_build_func,
        runtime = RUNTIME
    )
    runner = autotvm.LocalRunner(
        number = number,
        repeat = repeat,
        timeout = timeout,
        min_repeat_ms = min_repeat_ms,
        enable_cpu_cache_flush = True,
        module_loader = module_loader
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
            prefix = '[%s][Task: %2d/%2d] ' % (str(datetime.now().strftime('%Y/%m/%d %H:%M:%S')), i + 1, len(tasks))
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
            lib = relay.build(mod, target = TARGET, executor = EXECUTOR, runtime = RUNTIME, params = params)
else:
    with transform.PassContext(
        opt_level = opt_level, 
        config = {'tir.disable_vectorize': True}, 
        disabled_pass = ['AlterOpLayout']
    ):
        lib = relay.build(mod, target = TARGET, executor = EXECUTOR, runtime = RUNTIME, params = params)

# make C code file
if output_c_code:
    tvm.micro.export_model_library_format(lib, tar_file_path)
    with tarfile.open(tar_file_path, 'r:*') as tar_f:
        print('\n'.join(f' - {m.name}' for m in tar_f.getmembers()))

if zephyr_run:
    # flash to board
    template_project = pathlib.Path(tvm.micro.get_microtvm_template_projects('zephyr' if use_board else 'crt'))
    project_options = {
        'project_type': 'host_driven', 
        # 'cmsis_path' : '~/CMSIS',
        'zephyr_board': BOARD
    } if use_board else {}

    temp_dir = tvm.contrib.utils.tempdir(tvm_temp_path)
    generated_project_path = temp_dir / 'tvm_project'
    generated_project = tvm.micro.generate_project(
        template_project, lib, generated_project_path, project_options
    )
    generated_project.build()
    generated_project.flash()

    # exucute on the board
    with tvm.micro.Session(transport_context_manager = generated_project.transport()) as session:
        if executor_mode == 'graph':
            executor = tvm.micro.create_local_graph_executor(
                lib.get_graph_json(), session.get_system_lib(), session.device
            )
        elif executor_mode == 'aot':
            executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())

        executor.set_input(
            input_name, 
            img_data, 
            **lib.get_params()
        )

        total_time = 0.0
        for time in range(test_time):
            time_start = datetime.now().timestamp()
            executor.run()
            time_end = datetime.now().timestamp() # 計算 graph_mod 的執行時間
            total_time += time_end - time_start
            print('{0}. {1} -> {2}'.format(time+1, time_end - time_start, total_time))
        avg_time = total_time / test_time
        print('autoTVM use: {0}, avg spent {1}'.format(use_autoTVM, avg_time))

        tvm_output = executor.get_output(0).numpy()

        post.post_process(tvm_output[0], output_path, img_path, img_name)
