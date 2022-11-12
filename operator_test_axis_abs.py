import tvm

from tvm import relay, transform
from tvm.contrib import graph_executor

x = relay.var('x', shape=[2, 3, 4], dtype='int32')
output = relay.axis_abs(data = x, axis = 2, indice = 1) # axis 代表示幾個維度，indice 代表該維度中的第幾項
func = relay.Function([x], output)
mod = tvm.IRModule.from_expr(func)
print(mod)

with transform.PassContext(opt_level = 1):
    lib = relay.build(mod, target = {'cpu':tvm.target.Target('llvm')})

dev = tvm.device('llvm', 0)
module = graph_executor.GraphModule(lib['default'](dev))

data = [
    [[ -1,  -2,  -3,  -4], 
     [ -5,  -6,  -7,  -8], 
     [ -9, -10, -11, -12]],
    [[-13, -14, -15, -16], 
     [-17, -18, -19, -20], 
     [-21, -22, -23, -24]]
]
module.set_input('x', data)
module.run()
tvm_output = module.get_output(0).numpy()
print(tvm_output)