'''
使用模型： yolov5_704_p-0.9474_r-0.9408_map50-0.6404_192x192_ch1_ReLU-int8.tflite

在 stm32f429 上執行，包含 autoScheduler 功能

天鈺模型，已可以正確辨識天鈺的資料集

--- problem ---
新增 cmsis-nn 使用選擇，但 zephyr 若要支援 cmsis 則需要編譯 library 並提供路徑，目前還沒處理好
因此 zephyr 無法支援 cmsis 的情況下無法使用 autoScheduler 進行 tuning
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
from tvm import auto_scheduler, relay, transform
from tvm.auto_scheduler.task_scheduler import (LogEstimatedLatency,
                                               PrintTableInfo)
from tvm.contrib import graph_executor
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
use_autoScheduler = False
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

#------------------------------------------------------------------------------
# path setting
output_path = './test_outputs/fitipower_tflite_stm_autoScheduler_' + str(size)

model_path = './model/' + model_name
img_path = './img/' + img_name

original_relay_path = output_path + '/original_realy.txt'
original_params_path = output_path + '/original_params.txt'
converted_relay = output_path + '/converted_mod.txt'
cmsis_nn_relay = output_path + '/cmsis_nn_mod.txt'

records_path = output_path + '/autoScheduler.json'

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

#autoScheduler tuning
if use_autoScheduler:
    tasks, task_weights = auto_scheduler.extract_tasks(mod['main'], params, TARGET, opt_level=opt_level)

    for idx, task in enumerate(tasks):
        print('========== Task %d  (workload key: %s) ==========' % (idx, task.workload_key))
        print(task.compute_dag)

    module_loader = tvm.micro.AutoSchedulerModuleLoader(
        template_project_dir = str(pathlib.Path(tvm.micro.get_microtvm_template_projects('zephyr' if use_board else 'crt'))),
        zephyr_board = BOARD,
        west_cmd = 'west',
        verbose = False,
        project_type = 'host_driven'
    )
    local_rpc = auto_scheduler.LocalRPCMeasureContext(
        number = number,
        repeat = repeat, 
        timeout = timeout,
        min_repeat_ms = min_repeat_ms,
        enable_cpu_cache_flush=True,
        module_loader = module_loader
    )
    builder = auto_scheduler.LocalBuilder(
        disable_vectorize = True,
        build_func = tvm.micro.auto_scheduler_build_func,
        runtime = RUNTIME
    )
    runner = local_rpc.runner
    measure_callback = [auto_scheduler.RecordToFile(records_path)]
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials = trials,  # change this to 20000 to achieve the best performance
        early_stopping = early_stopping,
        builder = builder,
        runner = runner,
        measure_callbacks = measure_callback,
    )

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights, callbacks=[PrintTableInfo(), LogEstimatedLatency(output_path + '/total_latency.tsv')])
    
    if retune:
        if os.path.exists(records_path):
            os.remove(records_path)

        tuner.tune(tune_option)

# compile relay to tir
if use_autoScheduler:
    with auto_scheduler.ApplyHistoryBest(records_path):
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

if not using_cmsis_nn:
    # flash to board
    template_project = pathlib.Path(tvm.micro.get_microtvm_template_projects('zephyr' if use_board else 'crt'))
    project_options = {
        'project_type': 'host_driven', 
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
        print('autoScheduler use: {0}, avg spent {1}'.format(use_autoScheduler, avg_time))

        tvm_output = executor.get_output(0).numpy()

        post.post_process(tvm_output[0], output_path, img_path, img_name)
