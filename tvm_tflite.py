import os
import tarfile
from datetime import datetime

import cv2
import numpy
import tvm
from tvm import autotvm, relay, transform
from tvm.autotvm.tuner import XGBTuner
from tvm.contrib import graph_executor
from tvm.driver.tvmc.composite_target import get_codegen_by_target
from tvm.driver.tvmc.pass_config import parse_configs
from tvm.relay.backend import Executor


class Path:
    output_path = './test_outputs/fitipower@{0}' # model_name

    model_path = './model/{0}' # model_name
    img_path = './img/{0}' # img_name

    original_relay_path = '/original_realy.txt'
    original_params_path = '/original_params.txt'
    converted_relay = '/converted_mod.txt'
    cmsis_nn_relay = '/cmsis_nn_mod.txt'

    autoTVM_record = '/autoTVM@{0}@{1}.json'

    tar_file_path = '/c_code@{0}@{1}@{2}.tar' #model_name, tuner, executor_mode

    tvm_temp_path = '/home/yang880519/tvm_temp' # Warning：This folder will be removed every time.

class TargetInfo:
    executor_mode = 'graph'
    
    target = None
    executor = None

def path_init(model_name:str, img_name:str, executor_mode:str, transfer_layout:bool, use_autoTVM_log):
    Path.output_path = Path.output_path.format(model_name)

    Path.model_path = Path.model_path.format(model_name)
    Path.img_path = Path.img_path.format(img_name)

    Path.original_relay_path = Path.output_path + Path.original_relay_path
    Path.original_params_path = Path.output_path + Path.original_params_path
    Path.converted_relay = Path.output_path + Path.converted_relay
    Path.cmsis_nn_relay = Path.output_path + Path.cmsis_nn_relay

    Path.autoTVM_record = Path.output_path + Path.autoTVM_record.format(model_name, 'transLayout' if transfer_layout else 'oriLayout')

    if use_autoTVM_log:
        Path.tar_file_path = Path.output_path + Path.tar_file_path.format(model_name, 'autoTVM', executor_mode)
    else:
        Path.tar_file_path = Path.output_path + Path.tar_file_path.format(model_name, 'NoTuner', executor_mode)

    if not os.path.exists(Path.output_path):
        os.mkdir(Path.output_path)

def target_init(target, executor_mode):
    TargetInfo.executor_mode = executor_mode

    if target == 'llvm':
        TargetInfo.target = target
    else:
        raise RuntimeError('{0} is an unknown target.'.format(target))
    
    TargetInfo.executor = Executor(executor_mode)

def img_init(size:int):
    img_data = cv2.imread(Path.img_path, cv2.IMREAD_GRAYSCALE)
    img_data = cv2.resize(img_data, (size, size))
    img_data = numpy.array(img_data) - 128 # 量化到 int8 空間
    img_data = numpy.expand_dims(img_data, axis = (0, -1)).astype('int8')
    return img_data

def model_init(input_name:str, input_shape:set, input_dtype:str, opt_level:int, using_cmsis_nn:bool, transfer_layout:bool, IR_output:bool):
    model_buffer = open(Path.model_path, 'rb').read()

    try:
        import tflite
        model = tflite.Model.GetRootAsModel(model_buffer, 0)
    except AttributeError:
        import tflite.Model
        model = tflite.Model.Model.GetRootAsModel(model_buffer, 0)

    mod, params = relay.frontend.from_tflite(
        model = model,
        shape_dict = {input_name: input_shape},
        dtype_dict = {input_name: input_dtype}
    )

    if IR_output:
        print(mod, file = open(Path.original_relay_path, 'w'))
        print(params, file = open(Path.original_params_path, 'w'))

    if using_cmsis_nn:
        config = parse_configs(None)
        extra_targets = [{'name': 'cmsis-nn', 'opts': {'mcpu': 'cortex-m4'}, 'raw': 'cmsis-nn', 'is_tvm_target': False}]
        for codegen_from_cli in extra_targets:
                codegen = get_codegen_by_target(codegen_from_cli['name'])
                partition_function = codegen['pass_pipeline']

                if codegen['config_key'] is not None:
                    config[codegen['config_key']] = codegen_from_cli['opts']
                with transform.PassContext(config=config):
                    mod = partition_function(mod, params, mod_name='default', **codegen_from_cli['opts'])
    
        if IR_output:
            print(mod, file = open(Path.cmsis_nn_relay, 'w'))

    if transfer_layout:
        desired_layouts = {'qnn.conv2d': ['NCHW', 'default'], 'nn.max_pool2d':['NCHW', 'default'], 'image.resize2d':['NCHW']}
        seq = transform.Sequential([relay.transform.ConvertLayout(desired_layouts)]) #relay.transform.RemoveUnusedFunctions()
        with transform.PassContext(opt_level = opt_level):
            mod = seq(mod)

        if IR_output:
            print(mod, file = open(Path.converted_relay, 'w'))

    return mod, params

def init(img_name:str, size:int, 
         model_name:str, input_name:str, input_shape:set, input_dtype:str, 
         target:str, executor_mode:str, opt_level:int, using_cmsis_nn:bool, transfer_layout:bool, IR_output:bool, 
         use_autoTVM_log:bool):

    if executor_mode == 'aot':
        input_name = input_name.replace(':', '_')

    path_init(model_name, img_name, executor_mode, transfer_layout, use_autoTVM_log)

    target_init(target, executor_mode)

    img_data = img_init(size)

    mod, params = model_init(
        input_name, 
        input_shape, 
        input_dtype, 
        opt_level, 
        using_cmsis_nn, 
        transfer_layout, 
        IR_output
    )

    return input_name, img_data, mod, params

def autoTVM(mod, params, trials, number, repeat, timeout, min_repeat_ms, early_stopping):
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
        'tuning_records': Path.autoTVM_record,
    }
    if os.path.exists(tuning_option['tuning_records']):
        os.remove(tuning_option['tuning_records'])

    tasks = autotvm.task.extract_from_program(mod['main'], params = params, target = TargetInfo.target)
    assert len(tasks) > 0, 'No task for autoTVM tuning, please check your model.'

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

def tuning(tune_autoTVM, mod, params, trials, number, repeat, timeout, min_repeat_ms, early_stopping):
    if tune_autoTVM:
        autoTVM(mod, params, trials, number, repeat, timeout, min_repeat_ms, early_stopping)

def compile(mod, params, opt_level:int, output_c_code:bool, use_autoTVM_log:bool):
    assert TargetInfo.target and TargetInfo.executor, 'Target and Executor can not be \'None\'.'

    if use_autoTVM_log:
        assert os.path.exists(Path.autoTVM_record) is True, 'AutoTVM record is NOT FOUND.'
        dispatch_context = autotvm.apply_history_best(Path.autoTVM_record)
    else:
        dispatch_context = autotvm.DispatchContext.current

    with dispatch_context:
        with transform.PassContext(
            opt_level = opt_level, 
        ):
            lib = relay.build(
                mod, 
                target = TargetInfo.target, 
                executor = TargetInfo.executor, 
                params = params
            )

    # make C code file
    if output_c_code:
        tvm.micro.export_model_library_format(lib, Path.tar_file_path)
        with tarfile.open(Path.tar_file_path, 'r:*') as tar_f:
            print('\n'.join(f' - {m.name}' for m in tar_f.getmembers()))
    
    return lib

def run(lib, input_name, img_data, test_time):
    dev = tvm.device(TargetInfo.target, 0)

    if TargetInfo.executor_mode == 'graph':
        executor = graph_executor.GraphModule(lib['default'](dev))
    elif  TargetInfo.executor_mode == 'aot':
        temp_dir = tvm.contrib.utils.tempdir(Path.tvm_temp_path)
        test_so_path = temp_dir / 'test.so'
        lib.export_library(test_so_path, cc='gcc', options=['-std=c11', '-g3', '-O0'])
        loaded_mod = tvm.runtime.load_module(test_so_path)
        executor = tvm.runtime.executor.AotModule(loaded_mod['default'](dev))

    executor.set_input(input_name, img_data)

    total_time = 0.0
    for time in range(test_time):
        time_start = datetime.now().timestamp()
        executor.run()
        time_end = datetime.now().timestamp() # 計算 graph_mod 的執行時間
        total_time += time_end - time_start
        print('{0}. {1} -> {2}'.format(time+1, time_end - time_start, total_time))
    avg_time = total_time / test_time
    print('avg spent {0}'.format(avg_time))

    tvm_output = executor.get_output(0).numpy()
    return tvm_output[0]