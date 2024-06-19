export LD_LIBRARY_PATH=/opt/iree/lib: LD_LIBRARY_PATH
export PATH=/opt/iree/tools:$PATH

iree-compile --mlir-print-ir-after-all element_wise_TOSA.mlir