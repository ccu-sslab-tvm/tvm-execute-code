'''
使用模型： sine.tflite

在 stm32f429 上執行，包含 autoTVM 功能

目前已知問題： 無
'''

import json
import os
import pathlib
import tarfile
from datetime import datetime

import numpy
import tvm
from tvm import autotvm, relay, transform
from tvm.autotvm.tuner import XGBTuner
from tvm.relay.backend import Executor, Runtime

# path setting
output_folder_path = './test_outputs'
output_path = output_folder_path + '/sine_tflite_1_stm_autoTVM'
model_folder_path = './model'
tvm_temp_path = '/home/yang880519/tvm_temp' # Warning：This folder will be removed every time.

# TVM IR output setting
IR_output = True # Output relay & params or not
original_relay_path = output_path + '/original_realy.txt'
original_params_path = output_path + '/original_params.txt'

# board using setting
use_board = True # Use board or the simulator
simulator = 'qemu_x86'
physical_hw = 'stm32f429i_disc1'

# model using setting
model_path = model_folder_path + '/sine_model.tflite'
input_name = 'dense_4_input'
input_shape = (1,)
input_dtype = 'float32'

# optimize setting
opt_level = 3

# autoTVM setting
use_autoTVM = False
retune = False
number = 1
repeat = 1
trials = 1
early_stopping = 100
min_repeat_ms = 0 # since we're tuning on a CPU, can be set to 0
timeout = 100
records_path = output_path + '/autoTVM.json'

# make C code
output_c_code = True
tar_file_path = output_path + '/c_code.tar'

# executor mode
executor_mode = 'aot' # 'graph' or 'aot'

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

# load frontend model to TVM IR
mod, params = relay.frontend.from_tflite(
    model = model,
    shape_dict = {input_name: input_shape},
    dtype_dict = {input_name: input_dtype}
)

if IR_output:
    print(mod, file = open(original_relay_path, 'w'))
    print(params, file = open(original_params_path, 'w'))

# set target information
boards_file = pathlib.Path(tvm.micro.get_microtvm_template_projects('zephyr')) / 'boards.json'
with open(boards_file) as f:
    boards = json.load(f)

BOARD = physical_hw if use_board else simulator
TARGET = tvm.target.target.micro(boards[BOARD]['model'] if use_board else 'host')
RUNTIME = Runtime('crt', {'system-lib': True})
EXECUTOR = Executor('graph') if executor_mode == 'graph' else Executor("aot")

#autoTVM tuning
if use_autoTVM:
    moduler_loader = tvm.micro.AutoTvmModuleLoader(
        template_project_dir = pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr" if use_board else 'crt')),
        project_options = {
            "zephyr_board": BOARD,
            "west_cmd": "west",
            "verbose": False,
            "project_type": "host_driven"
        } if use_board else {
            "verbose": False
        }
    )
    builder = autotvm.LocalBuilder(
        build_kwargs = {"build_option": {"tir.disable_vectorize": True}},
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
        module_loader = moduler_loader
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
        tvm.nd.array(numpy.array([0.5], dtype = 'float32')), 
        **lib.get_params()
    )

    time_start = datetime.now()
    executor.run()
    time_end = datetime.now() # 計算 graph_mod 的執行時間
    print("spent {0}", time_end - time_start)

    tvm_output = executor.get_output(0).numpy()
    print(tvm_output)
