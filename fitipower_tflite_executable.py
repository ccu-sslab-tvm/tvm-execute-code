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
using_cmsis_nn = False # 基本常關
transfer_layout = False # 是否進行 layout 轉換
IR_output = True # Output relay & params or not

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
        using_cmsis_nn, 
        transfer_layout, 
        IR_output, 
        use_autoTVM_log, 
        use_autoScheduler_log, 
    )

    tvm.tuning(
        tune_autoTVM, 
        tune_autoScheduler, 
        mod, 
        params, 
        opt_level, 
        trials, 
        number, 
        repeat, 
        timeout, 
        min_repeat_ms, 
        early_stopping
    )

    lib = tvm.compile(
        mod, 
        params, 
        opt_level, 
        output_c_code, 
        use_autoTVM_log, 
        use_autoScheduler_log, 
    )

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
