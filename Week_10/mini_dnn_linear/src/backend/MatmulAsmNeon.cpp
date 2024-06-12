#include "MatmulAsmNeon.h"

extern "C" {
  void gemm_asm_asimd_64_64_64( float const * i_a,
                                float const * i_b,
                                float       * io_c );
}


at::Tensor mini_dnn::backend::MatmulAsmNeon::forward( at::Tensor i_x,
                                                      at::Tensor i_w ) {
  // get involved sizes
  Matmul::Sizes l_sizes = Matmul::getSizes( i_x,
                                            i_w );

  MINI_DNN_CHECK_EQ( l_sizes.bn, 64 );
  MINI_DNN_CHECK_EQ( l_sizes.bk, 64 );
  MINI_DNN_CHECK_EQ( l_sizes.bc, 64 );

  // prepare data for GEMM calls
  at::Tensor l_y = at::zeros( {l_sizes.kb, l_sizes.nb, l_sizes.bk, l_sizes.bn} );

  c10::IntArrayRef l_strides_a = i_x.strides();
  c10::IntArrayRef l_strides_b = i_w.strides();
  c10::IntArrayRef l_strides_c = l_y.strides();

  float * l_ptr_a = (float*) i_x.data_ptr();
  float * l_ptr_b = (float*) i_w.data_ptr();
  float * l_ptr_c = (float*) l_y.data_ptr();

  // TODO: Add OpenMP parallelization
  // loop over outer dimensions
  for( int64_t l_kb = 0; l_kb < l_sizes.kb; l_kb++ ) {
      for( int64_t l_nb = 0; l_nb < l_sizes.nb; l_nb++ ) {
        // TODO: compute offset in C

        for( int64_t l_cb = 0; l_cb < l_sizes.cb; l_cb++ ) {
          // TODO: compute offset in A

          // TODO: compute offset in B

          // TODO: call Neon assembly kernel
        }
      }
  }

  return l_y;
}