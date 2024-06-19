func.func @add( %lhs: tensor<3x2xf32>,
                %rhs: tensor<1x1xf32> ) -> tensor<3x2xf32> {
  %out = tosa.add %lhs, %rhs : (tensor<3x2xf32>, tensor<1x1xf32>) -> tensor<3x2xf32>
  return %out : tensor<3x2xf32>
}