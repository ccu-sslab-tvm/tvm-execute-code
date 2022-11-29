import tvm
from tvm import relay, transform
from tvm.contrib import graph_executor

x = relay.var('x', shape=[2, 3], dtype='int32')
y = relay.var('y', shape=[3, 2], dtype='int32')
output = relay.mat_mat_mul(mat1 = x, mat2 = y)
func = relay.Function([x, y], output)
mod = tvm.IRModule.from_expr(func)
print(mod)

with transform.PassContext(opt_level = 3):
    lib = relay.build(mod, target = {'cpu':tvm.target.Target('llvm')})

dev = tvm.device('llvm', 0)
module = graph_executor.GraphModule(lib['default'](dev))

mat1 = [
    [1, 2, 3],
    [4, 5, 6]
]

mat2 = [
    [1, 2],
    [3, 4],
    [5, 6]
]

module.set_input('x', mat1)
module.set_input('y', mat2)
module.run()
tvm_output = module.get_output(0).numpy()
print(tvm_output)