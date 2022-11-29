import os
from distutils.log import Log
from re import S

import numpy as np
import tvm
import tvm.relay as relay
from tvm import runtime
from tvm.contrib import graph_executor
from tvm.ir import Op
from tvm.ir.expr import RelayExpr
from tvm.ir.type import RelayRefType
from tvm.relay import create_executor

relay.RefWrite

def hete_test():
    from tvm.relay.expr_functor import ExprMutator

    class ScheduleConv2d(ExprMutator):
        def __init__(self, device):
            self.device = device
            super().__init__()
            self.hete_op = [Op.get("add")]

        def visit_call_1(self, expr):
            # print("*****visit_call_1*****")
            visit = super().visit_call(expr)
            self.memo_map[expr] = visit
            print("----visit_call_1 inside----")
            print(visit.op)
            print(expr.args)
            print(visit)
            print("@@@@@@@@@")
            # self.memo_map[expr] = expr
            if visit.op in self.hete_op:
                print(visit)
                print("---------------------------")
                return visit
            else :
                temp_device_copy = relay.device_copy(visit, tvm.cpu(),self.device)
                if temp_device_copy in self.memo_map.values(): #++
                    print(relay.device_copy(visit, tvm.cpu(),self.device))
                    print("in memo.map")
                    print("---------------------------")
                    return self.memo_map[temp_device_copy]
                else:
                    print(relay.device_copy(visit, tvm.cpu(),self.device))
                    self.memo_map[temp_device_copy] = temp_device_copy
                    print(self.memo_map)
                    return relay.device_copy(visit, tvm.cpu(),self.device)

        def ext_tmp(self, expr):
            print("------in ext_tmp------")
            print(expr.op)
            if expr.op in self.hete_op:
                print(expr)
                print("----------------")
                return expr
            else:
                temp_device_copy = relay.device_copy(expr, tvm.cpu(),self.device)
                if expr in self.memo_map.values(): #++
                    self.memo_map[temp_device_copy] = temp_device_copy
                    print(relay.device_copy(expr, tvm.cpu(),self.device))
                    print("in memo.map")
                    print("---------------------------")
                    return relay.device_copy(expr, tvm.cpu(),self.device)
                else:
                    print(relay.device_copy(expr, tvm.cpu(),self.device))
                    self.memo_map[temp_device_copy] = temp_device_copy
                    print(self.memo_map)
                    return relay.device_copy(expr, tvm.cpu(),self.device)

        def ext(self, expr):
            print("------in ext------")
            if expr in self.hete_op:
                print(expr)
                print("----------------")
                return expr
            else:
                print(relay.device_copy(expr, tvm.cpu(),self.device))
                print("------------------")
                return relay.device_copy(expr, tvm.cpu(),self.device)

        def imp(self, expr):
            print("------in imp------")
            print(expr)
            print("&&&&&&&&&")
            if expr.op in self.hete_op:
                temp_device_copy = relay.device_copy(expr, self.device, tvm.cpu())
                if temp_device_copy in self.memo_map.values():
                    print(relay.device_copy(expr, self.device, tvm.cpu()))
                    print("in memo.map")
                    print("---------------------")
                    return self.memo_map[temp_device_copy]
                else:
                    print(relay.device_copy(expr, self.device, tvm.cpu()))
                    self.memo_map[temp_device_copy] = temp_device_copy
                    print(str(temp_device_copy))
                    print(str(self.memo_map[temp_device_copy]))
                    print(self.memo_map)
                    print("---------------------")
                    return relay.device_copy(expr, self.device, tvm.cpu())
            else:
                print(expr)
                print("-----------------------")
                return expr


    def schedule_conv2d_on_gpu(expr):
        sched = ScheduleConv2d(tvm.cuda())
        return sched.visit(expr)

    x = relay.var("x", shape=(1, 10))
    y = relay.var("y", shape=(10, 10))
    add = relay.add(x,y)
    sqrt = relay.sqrt(add)
    log = relay.log(add)
    subtract = relay.subtract(sqrt,log)
    exp = relay.exp(subtract)
    add_1 = relay.add(exp, sqrt)

    func = relay.Function([x,y], add_1) # 這是relay function
    print(func)
    func = schedule_conv2d_on_gpu(func)
    print(func)
    mod = tvm.IRModule.from_expr(func) #這是ir module
    print(mod)

    x_data = np.array([[4,9,16,25,36,49,64,81,100,121]]).astype('float32')
    y_data = np.array([[4,9,16,25,36,49,64,81,100,121],
                        [4,9,16,25,36,49,64,81,100,121],
                        [4,9,16,25,36,49,64,81,100,121],
                        [4,9,16,25,36,49,64,81,100,121],
                        [4,9,16,25,36,49,64,81,100,121],
                        [4,9,16,25,36,49,64,81,100,121],
                        [4,9,16,25,36,49,64,81,100,121],
                        [4,9,16,25,36,49,64,81,100,121],
                        [4,9,16,25,36,49,64,81,100,121],
                        [4,9,16,25,36,49,64,81,100,121]]).astype('float32')

    print("============================")
    params = {"x": x_data, "y": y_data}


    with tvm.transform.PassContext(
        opt_level=0, config={"relay.fallback_device_type": tvm.cpu().device_type}
    ):
        lib = relay.build(mod, target={"cpu": tvm.target.Target("llvm"), "cuda": tvm.target.Target("cuda")})

    module = graph_executor.GraphModule(lib["default"](tvm.cpu(),tvm.cuda()))

    module.set_input(**params)
    module.run()
    out = module.get_output(0).asnumpy()

    print(out)
    print("============================")

    add = x_data + y_data
    sqrt = np.sqrt(add)
    log = np.log(add)
    subtract = sqrt - log
    exp = np.exp(subtract)
    add_1 = exp + sqrt
    print(add_1)

if __name__ == "__main__":
    hete_test()