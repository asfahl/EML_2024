linalg.generic #matmul_trait
  ins(%A, %B : memref<3x2xf32, stride_specification>,
               memref<2x3xf32, stride_specification>)
  outs(%C : memref<3x2xf32, stride_specification>)
  {other-optional-attributes} {
  ^bb0(%a: f32, %b: f32, %c: f32) :
    %d = arith.mulf %a, %b: f32
    %e = arith.addf %c, %d: f32
    linalg.yield %e : f32
}