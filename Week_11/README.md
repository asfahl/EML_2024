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
iree-check-module --device=local-task --module=compiled/add_TOSA.vmfb