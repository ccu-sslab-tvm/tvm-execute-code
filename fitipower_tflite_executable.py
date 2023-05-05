import tvm_tflite as tvm
from post_process import post_process_fitipower as post

# model setting
model_name = ''
size = 0
input_name = ''
input_shape = (1, size, size, 1)
input_dtype = ''

# input image setting
img_name = ''

# TVM IR output setting
use_cmsis_nn = False # 是否啟用 CMSIS-NN Operator 轉換
transfer_layout = False # 是否進行 layout 轉換
IR_output = False # Output relay & params or not

# runtime setting
target = 'llvm' # llvm, qemu_x86, stm32f429i_disc1, nucleo_h743zi
executor_mode = 'graph' # 'graph' or 'aot'
test_time = 1

# tuner setting
tune_autoTVM = False
tune_autoScheduler = False
number = 5
repeat = 3
trials = 20000
timeout = 120
min_repeat_ms = 0
early_stopping = 100

# tuner setting only for autoScheduler
auto_scheduler_alpha = 0.05
auto_scheduler_beta = 0.5
auto_scheduler_gamma = 0.5
auto_scheduler_bws = 1 #backward_window_size

# optimize setting
opt_level = 3
use_autoTVM_log = False
use_autoScheduler_log = False

# make C code
output_c_code = False

# image process setting
dequantance = [0, 0]
candidate = 0
class_num = 0
class_label = ['']

#--------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    input_name, img_data, mod, params = tvm.init(
        img_name, 
        size, 
        model_name, 
        input_name, 
        input_shape, 
        input_dtype, 
        target, 
        executor_mode, 
        opt_level, 
        use_cmsis_nn, 
        transfer_layout, 
        IR_output, 
        use_autoTVM_log, 
        use_autoScheduler_log, 
        output_c_code, 
    )

    tvm.tuning(
        tune_autoTVM, 
        tune_autoScheduler, 
        output_c_code, 
        mod, 
        params, 
        opt_level, 
        trials, 
        number, 
        repeat, 
        timeout, 
        min_repeat_ms, 
        early_stopping, 
        auto_scheduler_alpha, 
        auto_scheduler_beta, 
        auto_scheduler_gamma, 
        auto_scheduler_bws
    )

    lib = tvm.compile(
        mod, 
        params, 
        opt_level, 
        output_c_code, 
        use_autoTVM_log, 
        use_autoScheduler_log, 
    )

    if not output_c_code:
        output = tvm.run(
            lib, 
            input_name, 
            img_data, 
            test_time, 
        )

        post.post_process(
            output, 
            tvm.Path.output_path, 
            tvm.Path.img_path, 
            img_name, 
            size, 
            dequantance, 
            candidate, 
            class_num, 
            class_label, 
        )
