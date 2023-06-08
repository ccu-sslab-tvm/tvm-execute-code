import os
from datetime import datetime

import cv2
import numpy
import tvm
from tvm import auto_scheduler, autotvm, relay, transform
from tvm.auto_scheduler.task_scheduler import (LogEstimatedLatency,
                                               PrintTableInfo)
from tvm.autotvm.tuner import XGBTuner
from tvm.contrib import graph_executor
from tvm.relay.backend import Executor, Runtime
from tvm.runtime.executor import AotModule

from model_define import yolov5x_int8 as model_define

# input setting
input_index:int = 0 # for img or number

# output setting
IR_output:bool = 1
verbose_output:bool = 1

# ir setting
trans_layout:bool = 0

# runtime setting
target:str = 'llvm'
executor_mode:str = 'graph' #graph, aot
test_time:int = 1

# tuner setting
tune_autoTVM:bool = 0
tune_autoScheduler:bool = 0
use_previous:bool = 1
number:int = 2
repeat:int = 2
trials:int = 5000
timeout:int = 120
min_repeat_ms:int = 0
early_stopping:int = 500

# tuner setting only for autoScheduler
num_measures_per_round:int = 5
auto_scheduler_alpha:float = 0.2
auto_scheduler_beta:float = 2
auto_scheduler_gamma:float = 0.5
auto_scheduler_bws:int = 1
hardware_setting:dict = {
    'num_cores': 1,
    'vector_unit_bytes': 0,
    'cache_line_bytes': 32,
    'max_shared_memory_per_block': 0,
    'max_local_memory_per_block': 0,
    'max_threads_per_block': 0,
    'max_vthread_extent': 0,
    'warp_size': 0
}
include_simple_tasks:bool = 1

# optimize setting
opt_level:int = 3
use_autoTVM_log:bool = 0
use_autoScheduler_log:bool = 0

# -----------------------------------------------------------------------------
model_info = model_define()

try:
    img_selection = model_info.img[input_index]
    num_selection = None
except:
    img_selection = None
    num_selection = model_info.input_num[input_index]
assert (img_selection or num_selection) is not None, 'Image path is empty.'

output_path = f'./test_outputs/fitipower@{model_info.name}'

model_path = f'./model/{model_info.name}' # model_name
if img_selection is not None:
    img_path = f'./img/{img_selection}' # img_name
else:
    img_path = None

original_relay_path = output_path + '/original_mod.txt'
original_params_path = output_path + '/original_params.txt'
converted_relay_path = output_path + '/converted_mod.txt'

autoTVM_record = output_path \
                + f'/autoTVM@{target}' \
                + ('@trans' if trans_layout else '@ori') \
                + ('@NoCMSIS') \
                + '.json' 
autoScheduler_record = output_path \
                        + f'/autoScheduler@{target}' \
                        + ('@trans' if trans_layout else '@ori') \
                        + ('@NoCMSIS') \
                        + '.json'
autoScheduler_latency = output_path + '/total_latency.tsv'

tvm_temp_path = '/home/yang880519/tvm_temp' # Warning：This folder will be removed every time.

if not os.path.exists(output_path):
    os.mkdir(output_path)
# -----------------------------------------------------------------------------
# target define
runtime = Runtime('cpp')

if executor_mode == 'graph':
    executor = Executor('graph')
elif executor_mode == 'aot':
    executor = Executor('aot')

# image preprocess
if img_path is not None:
    if model_info.input_dtype == 'int8':
        img_data = cv2.imread(img_path, model_info.flags)
        img_data = cv2.resize(img_data, (model_info.img_height_width, model_info.img_height_width))
        img_data = numpy.array(img_data) - 128 # 量化到 int8 空間
        img_data = numpy.reshape(img_data, model_info.input_shape).astype('int8')
    elif model_info.input_dtype == 'float32':
        img_data = cv2.imread(img_path, model_info.flags)
        img_data = cv2.resize(img_data, (model_info.img_height_width, model_info.img_height_width))
        img_data = numpy.reshape(img_data, model_info.input_shape) / 255

if num_selection is not None:
    if model_info.input_dtype == 'int8':
        num_selection = numpy.array(num_selection) - 128

# load model
model_buffer = open(model_path, 'rb').read()

try:
    import tflite
    model = tflite.Model.GetRootAsModel(model_buffer, 0)
except AttributeError:
    import tflite.Model
    model = tflite.Model.Model.GetRootAsModel(model_buffer, 0)

mod, params = relay.frontend.from_tflite(
    model = model,
    shape_dict = {model_info.input_name: model_info.input_shape},
    dtype_dict = {model_info.input_name: model_info.input_dtype}
)

if IR_output:
    print(mod, file = open(original_relay_path, 'w'))
    print(params, file = open(original_params_path, 'w'))

if trans_layout:
    desired_layouts = {'qnn.conv2d': ['NCHW', 'default']}
    seq = transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
    with transform.PassContext(opt_level=opt_level):
        mod = seq(mod)

    if IR_output:
        print(mod, file = open(converted_relay_path, 'w'))

# tune autoTVM
if tune_autoTVM:
    tasks = autotvm.task.extract_from_program(mod, params = params, target = target)
    assert len(tasks) > 0, 'No task for autoTVM tuning, please check your model.'

    builder = autotvm.LocalBuilder()
    
    runner = autotvm.LocalRunner(
        number = number,
        repeat = repeat,
        timeout = timeout,
        min_repeat_ms = min_repeat_ms,
        enable_cpu_cache_flush = True
    )
    measure_option = autotvm.measure_option(
        builder = builder, 
        runner = runner
    )
    tuning_option = {
        'trials': trials,
        'early_stopping': early_stopping,
        'measure_option': measure_option,
        'tuning_records': autoTVM_record,
    }

    if os.path.exists(tuning_option['tuning_records']) and not use_previous:
        os.remove(tuning_option['tuning_records'])

    for i, task in enumerate(tasks):
        prefix = '[%s][Task: %2d/%2d] ' % (str(datetime.now().strftime('%Y/%m/%d %H:%M:%S')), i + 1, len(tasks))
        tuner_obj = XGBTuner(task, loss_type = 'rank')

        try:
            if use_previous:
                tuner_obj.load_history(autotvm.record.load_from_file(autoTVM_record))
        except Exception:
            Warning("The autoTVM tuning log does not exist.")

        tuner_obj.tune(
            n_trial = min(tuning_option['trials'], len(task.config_space)),
            early_stopping = tuning_option['early_stopping'],
            measure_option = tuning_option['measure_option'],
            callbacks = [
                autotvm.callback.progress_bar(tuning_option['trials'], prefix = prefix),
                autotvm.callback.log_to_file(tuning_option['tuning_records']),
            ],
        )

# tune autoScheduler
if tune_autoScheduler:
    if os.path.exists(autoScheduler_record) and not use_previous:
        os.remove(autoScheduler_record)

    tasks, task_weights = auto_scheduler.extract_tasks(
        mod = mod,
        params = params,
        target = target,
        include_simple_tasks = True,
        opt_level = opt_level
    )

    if verbose_output:
        for idx, task in enumerate(tasks):
            print('========== Task %d  (workload key: %s) ==========' % (idx, task.workload_key))
            print(task.compute_dag)

    builder = auto_scheduler.LocalBuilder()
    runner = auto_scheduler.LocalRunner(
        number = number,
        repeat = repeat, 
        timeout = timeout,
        min_repeat_ms = min_repeat_ms,
        enable_cpu_cache_flush=True
    )

    tuning_option = auto_scheduler.TuningOptions(
        num_measure_trials = trials, 
        early_stopping = early_stopping,
        num_measures_per_round = num_measures_per_round, 
        builder = builder,
        runner = runner,
        measure_callbacks = [auto_scheduler.RecordToFile(autoScheduler_record)],
    )

    tuner = auto_scheduler.TaskScheduler(
        tasks = tasks, 
        task_weights = task_weights, 
        load_log_file = autoScheduler_record if use_previous else None, 
        alpha = auto_scheduler_alpha, 
        beta = auto_scheduler_beta, 
        gamma = auto_scheduler_gamma, 
        backward_window_size = auto_scheduler_bws, 
        callbacks = [PrintTableInfo(), LogEstimatedLatency(autoScheduler_latency)]
    )

    tuner.tune(tuning_option)

# compile
config = {}
if use_autoScheduler_log:
    config['relay.backend.use_auto_scheduler'] = True

assert (use_autoTVM_log and use_autoScheduler_log) is not True, 'You can only choose either autoTVM or autoScheduler for compilation at a time.'
if use_autoTVM_log:
    assert os.path.exists(autoTVM_record) is True, 'AutoTVM record is NOT FOUND.'
    dispatch_context = autotvm.apply_history_best(autoTVM_record)
elif use_autoScheduler_log:
    assert os.path.exists(autoScheduler_record) is True, 'AutoScheduler record is NOT FOUND.'
    dispatch_context = auto_scheduler.ApplyHistoryBest(autoScheduler_record)
else:
    dispatch_context = autotvm.DispatchContext.current

with dispatch_context:
    with tvm.transform.PassContext(opt_level=opt_level, config=config):
        lib = relay.build(
            mod,
            target=target,
            executor=executor,
            runtime=runtime,
            params=params
        )

# execute
dev = tvm.device(target, 0)

if executor_mode == 'graph':
    model_executor = graph_executor.GraphModule(lib['default'](dev))
elif  executor_mode == 'aot':
    temp_dir = tvm.contrib.utils.tempdir(tvm_temp_path)
    test_so_path = temp_dir / 'test.so'
    lib.export_library(test_so_path, cc='gcc', options=['-std=c11', '-g3', '-O0'])
    loaded_mod = tvm.runtime.load_module(test_so_path)
    model_executor = AotModule(loaded_mod['default'](dev))

if img_selection is not None:
    model_executor.set_input(
        model_info.input_name, 
        img_data, 
        **lib.get_params()
    )
elif num_selection is not None:
    model_executor.set_input(
        model_info.input_name, 
        tvm.nd.array(numpy.array(num_selection, dtype=model_info.input_dtype)), 
        **lib.get_params()
    )

total_time = 0.0
for time in range(test_time):
    time_start = datetime.now().timestamp()
    model_executor.run()
    time_end = datetime.now().timestamp() # 計算 graph_mod 的執行時間
    total_time += time_end - time_start
    print('{0}. {1} -> {2}'.format(time+1, time_end - time_start, total_time))
avg_time = total_time / test_time
print('avg spent {0}'.format(avg_time))

tvm_output = model_executor.get_output(0).numpy()

# post process
try:
    model_info.post_process(
        tvm_output,
        output_path,
        img_path,
        img_selection
    )
except:
    print(model_info.dequantance[0] * (tvm_output + model_info.dequantance[1]))