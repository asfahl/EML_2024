#include <iostream>
#include <cmath>
#include <chrono>
#include "kernels/gemm_ref.h"

const unsigned int l_num_repetitions = 1000000;

extern "C" {
  void gemm_asm_asimd_64_64_64( float const * i_a,
                                float const * i_b,
                                float       * io_c );
}

float max_diff( float const * i_mat0,
                float const * i_mat1,
                unsigned int  i_m,
                unsigned int  i_n,
                unsigned int  i_ld ) {
  float l_maxDiff = 0;

  for( unsigned int l_m = 0; l_m < i_m; l_m++ ) {
    for( unsigned int l_n = 0; l_n < i_n; l_n++ ) {
      float l_diff = i_mat0[ l_n*i_ld + l_m ] - i_mat1[ l_n*i_ld + l_m ];
      l_diff = std::abs( l_diff );

      l_maxDiff = std::max( l_maxDiff, l_diff );
    }
  }

  return l_maxDiff;
}

int main() {
  std::chrono::high_resolution_clock::time_point l_tp0, l_tp1;
  std::chrono::duration<double> l_dur;
  double l_gFlops = 0;
  float l_maxDiff = 0;

  // allocate memory
  std::size_t l_size = 64*64;
  float * l_a = new float[ l_size ];
  float * l_b = new float[ l_size ];
  float * l_c = new float[ l_size ];
  float * l_cRef = new float[ l_size ];

  // init data
  srand48( time(NULL) );
  for( unsigned int l_id = 0; l_id < l_size; l_id++ ) {
    l_a[l_id] = (float) drand48();
  }
  for( unsigned int l_id = 0; l_id < l_size; l_id++ ) {
    l_b[l_id] = (float) drand48();
  }
  for( unsigned int l_id = 0; l_id < l_size; l_id++ ) {
    l_c[l_id] = (float) drand48();
  }
  for( unsigned int l_id = 0; l_id < l_size; l_id++ ) {
    l_cRef[l_id] = l_c[l_id];
  }

  /*
   * ASIMD: 64, 64, 64
   */
  std::cout << "testing gemm_asm_asimd_64_64_64 kernel" << std::endl;

  // run reference implementation
  gemm_ref_mnk( l_a,
                l_b,
                l_cRef,
                64,
                64,
                64,
                64,
                64,
                64 );

  // run assembly kernel
  gemm_asm_asimd_64_64_64( l_a,
                           l_b,
                           l_c );

  l_maxDiff = max_diff( l_cRef,
                        l_c,
                        64,
                        64,
                        64 );

  std::cout << "  maximum difference: " << l_maxDiff << "\n";

  // time asimd kernel
  l_tp0 = std::chrono::high_resolution_clock::now();
  for( unsigned int l_re = 0; l_re < l_num_repetitions; l_re++ ) {
    gemm_asm_asimd_64_64_64( l_a,
                             l_b,
                             l_c );
  }
  l_tp1 = std::chrono::high_resolution_clock::now();

  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );

  std::cout << "  duration: " << l_dur.count() << " seconds" << std::endl;
  l_gFlops  = l_num_repetitions;
  l_gFlops *= 64 * 64 * 64 * 2;
  l_gFlops *= 1.0E-9;
  l_gFlops /= l_dur.count();
  std::cout << "  GFLOPS: " << l_gFlops << std::endl;


  // free memory
  delete[] l_a;
  delete[] l_b;
  delete[] l_c;
  delete[] l_cRef;

  return EXIT_SUCCESS;
}
