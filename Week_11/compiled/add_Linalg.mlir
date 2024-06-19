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

