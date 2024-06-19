### Setting up IREE
export LD_LIBRARY_PATH=/opt/iree/lib: LD_LIBRARY_PATH
export PATH=/opt/iree/tools:$PATH

### Compiling
compiles element_wise_TOSA.mlir to add_TOSA.vmbf using the vmvx backend

...
iree-compile --iree-hal-target-backends=vmvx --mlir-print-ir-after-all element_wise_TOSA.mlir -o compiled/add_TOSA.vmfb
...

Output:
Way to verbous

### Execution and Testing
...
iree-run-module --module=compiled/add_TOSA.vmfb --device=local-task --function=add --input="3x2xf32=[1 2] [3 4] [5 6]" --input="1x1xf32=1"
...

Elementwise adds the 1x1 f32 vector [[1]] to the 3x2 f32 vector [[1, 2], [3, 4], [5, 6]] with the expected result 3x2 f32 [[2, 3], [4, 5], [6, 7]]

...
EXEC @add
result[0]: hal.buffer_view
3x2xf32=[2 3][4 5][6 7]
...

### Convert to linalg
...
iree-opt --pass-pipeline="builtin.module(func.func(tosa-to-linalg))" element_wise_TOSA.mlir -o compiled/add_Linalg.mlir
...

#### Output
...
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, 0)>
module {
  func.func @add(%arg0: tensor<3x2xf32>, %arg1: tensor<1x1xf32>) -> tensor<3x2xf32> {
    %0 = tensor.empty() : tensor<3x2xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<3x2xf32>, tensor<1x1xf32>) outs(%0 : tensor<3x2xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<3x2xf32>
    return %1 : tensor<3x2xf32>
  }
}
...

convert tensors to buffers

convert to parallel loops
...
iree-opt --pass-pipeline="builtin.module(func.func(convert-linalg-to-parallel-loops))" compiled/add_Linalg.mlir -o compiled/add_ParaLoop.mlir
...
not seeing much difference