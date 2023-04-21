import json
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
from tvm.contrib import graph_executor
from tvm.driver.tvmc.composite_target import get_codegen_by_target
from tvm.driver.tvmc.pass_config import parse_configs
from tvm.relay.backend import Executor, Runtime
from tvm.relay.op.contrib import cmsisnn

computer_target_list = {'llvm'}
zephyr_qemu_list = {'qemu_x86'}
zephyr_board_list = {'stm32f429i_disc1', 'nucleo_h743zi'}

class Path:
    output_path = './test_outputs/fitipower@{0}' # model_name

    model_path = './model/{0}' # model_name
    img_path = './img/{0}' # img_name

    original_relay_path = '/original_realy.txt'
    original_params_path = '/original_params.txt'
    converted_relay = '/converted_mod.txt'
    cmsis_nn_relay = '/cmsis_nn_mod.txt'

    autoTVM_record = '/autoTVM@{0}@{1}@{2}@{3}.json' # target_name, layout, executor_mode, CMSIS
    autoScheduler_record = '/autoScheduler@{0}@{1}@{2}@{3}.json' # target_name, layout, executor_mode, CMSIS
    autoScheduler_latency = '/total_latency.tsv'

    tar_file_path = '/c_code@{0}@{1}@{2}@{3}@{4}.tar' # target_name, tuner, executor_mode, layout, CMSIS

    tvm_temp_path = '/home/yang880519/tvm_temp' # Warning：This folder will be removed every time.

class TargetInfo:
    target_name = None
    executor_mode = 'graph'
    
    target = None
    runtime = None
    executor = None

def path_init(model_name:str, img_name:str, use_cmsis_nn:bool, transfer_layout:bool, use_autoTVM_log:bool, use_autoScheduler_log:bool):
    Path.output_path = Path.output_path.format(model_name)

    Path.model_path = Path.model_path.format(model_name)
    Path.img_path = Path.img_path.format(img_name)

    Path.original_relay_path = Path.output_path + Path.original_relay_path
    Path.original_params_path = Path.output_path + Path.original_params_path
    Path.converted_relay = Path.output_path + Path.converted_relay
    Path.cmsis_nn_relay = Path.output_path + Path.cmsis_nn_relay

    Path.autoTVM_record = Path.output_path + Path.autoTVM_record.format(
        TargetInfo.target_name, 
        'transLayout' if transfer_layout else 'oriLayout', 
        TargetInfo.executor_mode, 
        'CMSIS' if use_cmsis_nn else 'NoCMSIS'
    )
    Path.autoScheduler_record = Path.output_path + Path.autoScheduler_record.format(
        TargetInfo.target_name, 
        'transLayout' if transfer_layout else 'oriLayout', 
        TargetInfo.executor_mode, 
        'CMSIS' if use_cmsis_nn else 'NoCMSIS'
    )
    Path.autoScheduler_latency = Path.output_path + Path.autoScheduler_latency

    if use_autoTVM_log:
        Path.tar_file_path = Path.output_path + Path.tar_file_path.format(
            TargetInfo.target_name, 
            'autoTVM', 
            TargetInfo.executor_mode, 
            'transLayout' if transfer_layout else 'oriLayout', 
            'CMSIS' if use_cmsis_nn else 'NoCMSIS'
        )
    elif use_autoScheduler_log:
        Path.tar_file_path = Path.output_path + Path.tar_file_path.format(
            TargetInfo.target_name, 
            'autoScheduler', 
            TargetInfo.executor_mode, 
            'transLayout' if transfer_layout else 'oriLayout', 
            'CMSIS' if use_cmsis_nn else 'NoCMSIS'
        )
    else:
        Path.tar_file_path = Path.output_path + Path.tar_file_path.format(
            TargetInfo.target_name, 
            'NoTuner', 
            TargetInfo.executor_mode, 
            'transLayout' if transfer_layout else 'oriLayout', 
            'CMSIS' if use_cmsis_nn else 'NoCMSIS'
        )

    if not os.path.exists(Path.output_path):
        os.mkdir(Path.output_path)

def target_init(target, executor_mode):
    TargetInfo.target_name = target
    TargetInfo.executor_mode = executor_mode

    if target in computer_target_list:
        TargetInfo.target = target
        TargetInfo.runtime = Runtime('cpp')
    elif target in (zephyr_qemu_list | zephyr_board_list):
        with open(pathlib.Path(tvm.micro.get_microtvm_template_projects('zephyr')) / 'boards.json') as f:
            boards = json.load(f)
        TargetInfo.target = tvm.target.target.micro(boards[target]['model'] if target in zephyr_board_list else 'host')
        TargetInfo.runtime = Runtime('crt', {'system-lib': True})
    else:
        raise RuntimeError('{0} is an unknown target.'.format(target))
    
    if executor_mode == 'graph':
        TargetInfo.executor = Executor('graph')
    elif executor_mode == 'aot':
        TargetInfo.executor = Executor("aot", {"unpacked-api": True, "interface-api": "c"})
    else:
        raise RuntimeError('Unknown Executor')


def img_init(size:int):
    img_data = cv2.imread(Path.img_path, cv2.IMREAD_GRAYSCALE)
    img_data = cv2.resize(img_data, (size, size))
    img_data = numpy.array(img_data) - 128 # 量化到 int8 空間
    img_data = numpy.expand_dims(img_data, axis = (0, -1)).astype('int8')
    return img_data

def model_init(input_name:str, input_shape:set, input_dtype:str, opt_level:int, use_cmsis_nn:bool, transfer_layout:bool, IR_output:bool):
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

    if use_cmsis_nn:
        if TargetInfo.target_name in (zephyr_qemu_list | zephyr_board_list):
            config = {'tir.disable_vectorize': True}
            config['relay.ext.cmsisnn.options'] = {'mcpu': TargetInfo.target.mcpu}
            with transform.PassContext(config=config):
                mod = cmsisnn.partition_for_cmsisnn(mod, params, mcpu=TargetInfo.target.mcpu)
        
            if IR_output:
                print(mod, file = open(Path.cmsis_nn_relay, 'w'))
        else:
            raise RuntimeError(f'{TargetInfo.target_name} is not supported CMSIS-NN library')

    if transfer_layout:
        desired_layouts = {'qnn.conv2d': ['NCHW', 'default'], 'nn.max_pool2d':['NCHW', 'default'], 'image.resize2d':['NCHW']}
        seq = transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
        with transform.PassContext(opt_level = opt_level):
            mod = seq(mod)

        if IR_output:
            print(mod, file = open(Path.converted_relay, 'w'))

    return mod, params

def init(img_name:str, size:int, 
         model_name:str, input_name:str, input_shape:set, input_dtype:str, 
         target:str, executor_mode:str, opt_level:int, use_cmsis_nn:bool, transfer_layout:bool, IR_output:bool, 
         use_autoTVM_log:bool, use_autoScheduler_log:bool):

    if executor_mode == 'aot':
        input_name = input_name.replace(':', '_')
    
    assert (use_autoTVM_log and use_autoScheduler_log) is not True, 'It can only use autoTVM or autoScheduler tuning log to compile at one time.'

    target_init(target, executor_mode)

    path_init(model_name, img_name, use_cmsis_nn, transfer_layout, use_autoTVM_log, use_autoScheduler_log)

    img_data = img_init(size)

    mod, params = model_init(
        input_name, 
        input_shape, 
        input_dtype, 
        opt_level, 
        use_cmsis_nn, 
        transfer_layout, 
        IR_output
    )

    return input_name, img_data, mod, params

def autoTVM_option(trials, number, repeat, timeout, min_repeat_ms, early_stopping):
    if TargetInfo.target_name in computer_target_list:
        builder = autotvm.LocalBuilder()
    elif TargetInfo.target_name in (zephyr_qemu_list | zephyr_board_list):
        module_loader = tvm.micro.AutoTvmModuleLoader(
            template_project_dir = pathlib.Path(
                tvm.micro.get_microtvm_template_projects(
                    'zephyr' if TargetInfo.target_name in zephyr_board_list else 'crt'
                )
            ), 
            project_options = {
                'zephyr_board': TargetInfo.target_name,
                'west_cmd': 'west',
                'verbose': False,
                'project_type': 'host_driven'
            } if TargetInfo.target_name in zephyr_board_list else {
                'verbose': False,
            }
        )
        builder = autotvm.LocalBuilder(
            build_kwargs = {'build_option': {'tir.disable_vectorize': True}},
            do_fork = True,
            build_func = tvm.micro.autotvm_build_func,
            runtime = TargetInfo.runtime
        )
    else:
        raise RuntimeError('AutoTVM setting failed, please check your target.')
    
    runner = autotvm.LocalRunner(
        number = number,
        repeat = repeat,
        timeout = timeout,
        min_repeat_ms = min_repeat_ms,
        enable_cpu_cache_flush = True,
        module_loader = module_loader if TargetInfo.target_name in (zephyr_qemu_list | zephyr_board_list) else None
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
    return tuning_option

def autoTVM(mod, params, trials, number, repeat, timeout, min_repeat_ms, early_stopping):
    tuning_option = autoTVM_option(trials, number, repeat, timeout, min_repeat_ms, early_stopping)

    if os.path.exists(tuning_option['tuning_records']):
        os.remove(tuning_option['tuning_records'])

    tasks = autotvm.task.extract_from_program(mod['main'], params = params, target = TargetInfo.target)
    assert len(tasks) > 0, 'No task for autoTVM tuning, please check your model.'

    for i, task in enumerate(tasks):
        prefix = '[%s][Task: %2d/%2d] ' % (str(datetime.now().strftime('%Y/%m/%d %H:%M:%S')), i + 1, len(tasks))
        tuner_obj = XGBTuner(task, loss_type = 'rank')
        tuner_obj.tune(
            n_trial = min(tuning_option['trials'], len(task.config_space)),
            early_stopping = tuning_option['early_stopping'],
            measure_option = tuning_option['measure_option'],
            callbacks = [
                autotvm.callback.progress_bar(tuning_option['trials'], prefix = prefix),
                autotvm.callback.log_to_file(tuning_option['tuning_records']),
            ],
        )

def autoScheduler_option(trials, number, repeat, timeout, min_repeat_ms, early_stopping):
    if TargetInfo.target_name in computer_target_list:
        builder = auto_scheduler.LocalBuilder()
        runner = auto_scheduler.LocalRunner(
            number = number,
            repeat = repeat, 
            timeout = timeout,
            min_repeat_ms = min_repeat_ms,
            enable_cpu_cache_flush=True
        )
    elif TargetInfo.target_name in zephyr_qemu_list:
        raise RuntimeError('AutoScheduler module loader is not support qemulator target now.')
    elif TargetInfo.target_name in zephyr_board_list:
        module_loader = tvm.micro.AutoSchedulerModuleLoader(
            template_project_dir = str(pathlib.Path(tvm.micro.get_microtvm_template_projects('zephyr'))),
            zephyr_board = TargetInfo.target_name,
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
            timeout = timeout, 
            disable_vectorize = True,
            build_func = tvm.micro.auto_scheduler_build_func,
            runtime = TargetInfo.runtime
        )
        runner = local_rpc.runner
    else:
        raise RuntimeError('AutoTVM setting failed, please check your target.')
    
    tuning_option = auto_scheduler.TuningOptions(
        num_measure_trials = trials, 
        early_stopping = early_stopping,
        builder = builder,
        runner = runner,
        measure_callbacks = [auto_scheduler.RecordToFile(Path.autoScheduler_record)],
    )
    return local_rpc if 'local_rpc' in locals() else None, tuning_option

def autoScheduler(mod, params, opt_level, trials, number, repeat, timeout, min_repeat_ms, early_stopping):
    tasks, task_weights = auto_scheduler.extract_tasks(mod['main'], params, TargetInfo.target, opt_level=opt_level)

    for idx, task in enumerate(tasks):
        print('========== Task %d  (workload key: %s) ==========' % (idx, task.workload_key))
        print(task.compute_dag)

    _, tuning_option = autoScheduler_option(trials, number, repeat, timeout, min_repeat_ms, early_stopping)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights, callbacks=[PrintTableInfo(), LogEstimatedLatency(Path.autoScheduler_latency)])

    if os.path.exists(Path.autoScheduler_record):
        os.remove(Path.autoScheduler_record)

    tuner.tune(tuning_option)

def tuning(tune_autoTVM, tune_autoScheduler, mod, params, opt_level, trials, number, repeat, timeout, min_repeat_ms, early_stopping):
    if tune_autoTVM:
        try:
            autoTVM(mod, params, trials, number, repeat, timeout, min_repeat_ms, early_stopping)
        except Exception as e:
            print('autoTVM tuning failed:')
            print(e)
            if os.path.exists(Path.autoTVM_record):
                os.remove(Path.autoTVM_record)

    if tune_autoScheduler:
        try:
            autoScheduler(mod, params, opt_level, trials, number, repeat, timeout, min_repeat_ms, early_stopping)
        except Exception as e:
            print('autoScheduler tuning failed:')
            print(e)
            if os.path.exists(Path.autoScheduler_record):
                os.remove(Path.autoScheduler_record)
            if os.path.exists(Path.autoScheduler_latency):
                os.remove(Path.autoScheduler_latency)

def compile(mod, params, opt_level:int, output_c_code:bool, use_autoTVM_log:bool, use_autoScheduler_log:bool):
    assert TargetInfo.target and TargetInfo.executor, 'Target and Executor can not be \'None\'.'

    if use_autoTVM_log:
        assert os.path.exists(Path.autoTVM_record) is True, 'AutoTVM record is NOT FOUND.'
        dispatch_context = autotvm.apply_history_best(Path.autoTVM_record)
    elif use_autoScheduler_log:
        assert os.path.exists(Path.autoScheduler_record) is True, 'AutoScheduler record is NOT FOUND.'
        dispatch_context = auto_scheduler.ApplyHistoryBest(Path.autoScheduler_record)
    else:
        dispatch_context = autotvm.DispatchContext.current
    
    config = {}
    if TargetInfo.target_name in (zephyr_qemu_list | zephyr_board_list):
        config['tir.disable_vectorize'] = True
    if use_autoScheduler_log:
        config['relay.backend.use_auto_scheduler'] = True

    with dispatch_context:
        with transform.PassContext(
            opt_level = opt_level, 
            config = config, 
        ):
            lib = relay.build(
                mod, 
                target = TargetInfo.target, 
                executor = TargetInfo.executor, 
                runtime = TargetInfo.runtime, 
                params = params
            )

    # make C code file
    if output_c_code:
        tvm.micro.export_model_library_format(lib, Path.tar_file_path)
        with tarfile.open(Path.tar_file_path, 'r:*') as tar_f:
            print('\n'.join(f' - {m.name}' for m in tar_f.getmembers()))
    
    return lib

def run_computer(lib, input_name, img_data, test_time):
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

def run_zephyr(lib, input_name, img_data, test_time):
    # flash to board
    template_project = pathlib.Path(
        tvm.micro.get_microtvm_template_projects(
            'zephyr' if TargetInfo.target_name in zephyr_board_list else 'crt'
        )
    )
    project_options = {
        'project_type': 'host_driven', #host_driven, aot_standalone_demo
        'zephyr_board': TargetInfo.target_name, 
    } if TargetInfo.target_name in zephyr_board_list else {}

    temp_dir = tvm.contrib.utils.tempdir(Path.tvm_temp_path)
    generated_project_path = temp_dir / 'tvm_project'
    generated_project = tvm.micro.generate_project(
        template_project, lib, generated_project_path, project_options
    )
    generated_project.build()
    generated_project.flash()

    with tvm.micro.Session(transport_context_manager = generated_project.transport()) as session:
        if TargetInfo.executor_mode == 'graph':
            executor = tvm.micro.create_local_graph_executor(
                lib.get_graph_json(), session.get_system_lib(), session.device
            )
        elif TargetInfo.executor_mode == 'aot':
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
        print('avg spent {0}'.format(avg_time))

        tvm_output = executor.get_output(0).numpy()
    return tvm_output[0]

def run(lib, input_name, img_data, test_time):
    if TargetInfo.target_name in computer_target_list:
        return run_computer(lib, input_name, img_data, test_time)
    elif TargetInfo.target_name in (zephyr_qemu_list | zephyr_board_list):
        return run_zephyr(lib, input_name, img_data, test_time)
    else:
        raise RuntimeError('Run tvm failed, please check your target.')