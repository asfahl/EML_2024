#ifndef MINI_DNN_BACKEND_MATMUL_ASM_NEON_H
#define MINI_DNN_BACKEND_MATMUL_ASM_NEON_H

#include "Matmul.hpp"
#include <ATen/ATen.h>

namespace mini_dnn {
  namespace backend {
    class MatmulAsmNeon;
  }
}

/**
 * Matmul backend using a Neon assembly kernel.
 **/
class mini_dnn::backend::MatmulAsmNeon: public Matmul {
  private:
  public:
    /**
     * Perform the forward pass, i.e., Y = XW.
     *
     * @param i_x matrix X.
     * @param i_w matrix W.
     * @return output of the matmul, i.e., Y.
     **/
    at::Tensor forward( at::Tensor i_x,
                        at::Tensor i_w );
};

#endif