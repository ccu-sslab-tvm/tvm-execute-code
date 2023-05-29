import os
import pathlib
import tarfile
from datetime import datetime

import cv2
import numpy
import tvm
from tvm import auto_scheduler, autotvm, relay, transform
from tvm.auto_scheduler.task_scheduler import (LogEstimatedLatency,
                                               PrintTableInfo)
from tvm.autotvm.tuner import XGBTuner
from tvm.contrib.utils import tempdir
from tvm.micro.testing import get_target
from tvm.relay.backend import Executor, Runtime
from tvm.relay.op.contrib import cmsisnn

from model_define import choose_the_model as model_define

# input setting
input_index:int = 0 # for img or number

# output setting
IR_output:bool = 1
verbose_output:bool = 1

# ir setting
use_cmsis:bool = 0
trans_layout:bool = 0

# runtime setting
board_name:str = 'stm32f429i_disc1'
executor_mode:str = 'aot' #graph, aot
test_time:int = 1

# tuner setting
tune_autoTVM:bool = 0
tune_autoScheduler:bool = 0
use_previous:bool = 1
number:int = 5
repeat:int = 3
trials:int = 20000
timeout:int = 120
min_repeat_ms:int = 0
early_stopping:int = 100

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

# project_type for Zephyr
project_type:str = 'host_driven'

# make C code
output_c_code:bool = 0

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

img_rawData_path = output_path + '/rawData.c'

original_relay_path = output_path + '/original_mod.txt'
original_params_path = output_path + '/original_params.txt'
converted_relay_path = output_path + '/converted_mod.txt'
cmsis_relay_path = output_path + '/cmsis_mod.txt'

autoTVM_record = output_path \
                + f'/autoTVM@{board_name}' \
                + ('@trans' if trans_layout else '@ori') \
                + ('@CMSIS' if use_cmsis else '@NoCMSIS') \
                + '.json' 
autoScheduler_record = output_path \
                        + f'/autoScheduler@{board_name}' \
                        + ('@trans' if trans_layout else '@ori') \
                        + ('@CMSIS' if use_cmsis else '@NoCMSIS') \
                        + '.json'
autoScheduler_latency = output_path + '/total_latency.tsv'

tar_file_path = output_path \
                + f'/c_code@{board_name}' \
                + '@NoneTuner' \
                + '@aot' \
                + ('@trans' if trans_layout else '@ori') \
                + ('@CMSIS' if use_cmsis else '@NoCMSIS') \
                + '.tar'

tvm_temp_path = '/home/yang880519/tvm_temp' # Warning：This folder will be removed every time.

if not os.path.exists(output_path):
    os.mkdir(output_path)

#----------------------------------------------------------------------------------------------------------------------
# target define
target = get_target('zephyr', board_name)
if output_c_code and (executor_mode == 'aot'):
    runtime = Runtime('crt')
else:
    runtime = Runtime('crt', {'system-lib': True})

if executor_mode == 'graph':
    if output_c_code:
        executor = Executor('graph', {"link-params": True})
    else:
        executor = Executor('graph')
elif executor_mode == 'aot':
    if output_c_code:
        executor = Executor('aot', {"unpacked-api": True, "interface-api": "c"})
    else:
        executor = Executor('aot')

# image preprocess
if img_path is not None:
    if model_info.input_dtype == 'int8':
        img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (model_info.img_height_width, model_info.img_height_width))
        img_data = numpy.array(img_data) - 128 # 量化到 int8 空間
        img_data = numpy.expand_dims(img_data, axis = (0, -1)).astype('int8')
    elif model_info.input_dtype == 'float32':
        img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (model_info.img_height_width, model_info.img_height_width))
        img_data = numpy.expand_dims(img_data, axis = (0, -1)) / 255

    if verbose_output:
        rawData = img_data.reshape(model_info.img_height_width*model_info.img_height_width)
        count = 0
        str_rawDara = '\r\n\t'
        for i in rawData:
            str_rawDara += str(i) + ', '
            if (count+1) % model_info.img_height_width == 0:
                str_rawDara += '\r\n\t'
            count += 1
        print('int8_t raw_data[] = {' + str_rawDara + '};', file=open(img_rawData_path, 'w'))

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

assert (use_cmsis and trans_layout) is not True, 'You can only choose cmsis or layout transfer at one time.'

if use_cmsis:
    mod = cmsisnn.partition_for_cmsisnn(mod, params, mcpu=target.mcpu)

    if IR_output:
        print(mod, file = open(cmsis_relay_path, 'w'))

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

    project_options = {
        'west_cmd': 'west',
        'board': board_name,
        'project_type': project_type,
        'config_main_stack_size': 8192,
        'workspace_size_bytes': 168*1024,
        'config_memc': 1, #aot & graph need
        'config_sys_heap_big_only': 1, #graph need
        'verbose': 1,
    }
    module_loader = tvm.micro.AutoTvmModuleLoader(
        template_project_dir = str(pathlib.Path(tvm.micro.get_microtvm_template_projects('zephyr'))),
        project_options = project_options
    )
    builder = autotvm.LocalBuilder(
        build_kwargs = {'build_option': {'tir.disable_vectorize': True}},
        do_fork = True,
        build_func = tvm.micro.autotvm_build_func,
        runtime = runtime
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

    hardware_params = auto_scheduler.HardwareParams(
        num_cores = hardware_setting['num_cores'],
        vector_unit_bytes = hardware_setting['vector_unit_bytes'],
        cache_line_bytes = hardware_setting['cache_line_bytes'],
        max_shared_memory_per_block = hardware_setting['max_shared_memory_per_block'],
        max_local_memory_per_block = hardware_setting['max_local_memory_per_block'],
        max_threads_per_block = hardware_setting['max_threads_per_block'],
        max_vthread_extent = hardware_setting['max_vthread_extent'],
        warp_size = hardware_setting['warp_size'],
    )

    tasks, task_weights = auto_scheduler.extract_tasks(
        mod = mod, 
        params = params, 
        target = target, 
        hardware_params = hardware_params, 
        include_simple_tasks = include_simple_tasks, 
        opt_level = opt_level
    )

    if verbose_output:
        for idx, task in enumerate(tasks):
            print('========== Task %d  (workload key: %s) ==========' % (idx, task.workload_key))
            print(task.compute_dag)

    project_options = {
        'west_cmd': 'west',
        'board': board_name,
        'project_type': project_type,
        'config_main_stack_size': 8192,
        'workspace_size_bytes': 168*1024,
        'config_memc': 1, #aot & graph need
        'config_sys_heap_big_only': 1, #graph need
        'verbose': 1,
    }
    module_loader = tvm.micro.AutoSchedulerModuleLoader(
        template_project_dir = str(pathlib.Path(tvm.micro.get_microtvm_template_projects('zephyr'))),
        project_options = project_options
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
        timeout = timeout, 
        disable_vectorize = True,
        build_func = tvm.micro.auto_scheduler_build_func,
        runtime = runtime
    )
    tuning_option = auto_scheduler.TuningOptions(
        num_measure_trials = trials, 
        early_stopping = early_stopping,
        num_measures_per_round = num_measures_per_round, 
        builder = builder,
        runner = local_rpc.runner,
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
config = {
    'tir.disable_vectorize': True
}
if use_cmsis:
    config['relay.ext.cmsisnn.options'] = {'mcpu': target.mcpu}
if output_c_code and (executor_mode == 'aot'):
    config['tir.usmp.enable'] = True
    config['tir.usmp.algorithm'] = 'hill_climb'
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

# make C code file
if output_c_code:
    tvm.micro.export_model_library_format(lib, tar_file_path)
    if verbose_output:
        with tarfile.open(tar_file_path, 'r:*') as tar_f:
            print('\n'.join(f' - {m.name}' for m in tar_f.getmembers()))

# execute
template_project = pathlib.Path(
    tvm.micro.get_microtvm_template_projects('zephyr')
)
project_options = {
    'project_type': project_type,
    'board': board_name,
    'config_main_stack_size': 8192,
    'workspace_size_bytes': 168*1024,
    'config_memc': 1, #aot & graph need
    'config_sys_heap_big_only': 1, #graph need
    'verbose': 1,
}
if use_cmsis:
    project_options['cmsis_path'] = '~/CMSIS_5'
    project_options['compile_definitions'] = [f'-DCOMPILE_WITH_CMSISNN=1']

temp_dir = tempdir(tvm_temp_path)
generated_project_path = temp_dir / 'tvm_project'
generated_project = tvm.micro.generate_project(
    template_project, lib, generated_project_path, project_options
)
generated_project.build()
generated_project.flash()

with tvm.micro.Session(transport_context_manager = generated_project.transport()) as session:
    if executor_mode == 'graph':
        model_executor = tvm.micro.create_local_graph_executor(
            lib.get_graph_json(), session.get_system_lib(), session.device
        )
    elif executor_mode == 'aot':
        model_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())

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