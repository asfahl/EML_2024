
linalg.generic  @matrix_multiplication = func() {
  %A_tile : memref<?x, 32> = arange<i32>(0, A.shape[0]) 
  %B_tile : memref<32, ?y> = arange<i32>(0, B.shape[1]) 

  linalg.for_range %A_iter = affine_map<(affine_dim<0>) -> affine_dim<?x>> { a: affine_value 0 } {
    linalg.for_range %B_iter = affine_map<(affine_dim<1>) -> affine_dim<?y>> { b: affine_value 0 } {
      let i = %A_iter + a;
      let j = %B_iter + b;

      linalg.assign %C[i, j] = linalg.matmulOP(%A_tile[i], %B_tile[j]);
    }
  }
  return %C 
}
