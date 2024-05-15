import torch
import triton
import triton.language as tl

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

@triton.autotune(
    configs=[triton.Config( { 'SIZE_BLOCK_L': 1, 'SIZE_BLOCK_M': 64, 'SIZE_BLOCK_N': 64, 'SIZE_BLOCK_K': 16 },
                                 num_stages = 3, num_warps = 2 )],
    key=['size_a_l', 'size_b_l', 'size_m', 'size_n', 'size_k']
    )


@triton.jit
def matmul_kernel(
    #Matrix pointers
    a_ptr, b_ptr, _c_ptr,
    #Matrix dimensions
    size_m, size_n, size_k, size_a_l, size_b_l,
    #strides
    stride_a_l, stride_a_m, stride_a_k, stride_b_l, stride_b_k, stride_b_n, stride_c_l, stride_c_m, stride_c_n,
    # Meta-Parameter
    BLOCK_SIZE_L: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
    ):
    pid = tl.program_id(axis=0)

    # derive location of pid
    num_blocks_m = triton.language.cdiv( size_m, BLOCK_SIZE_M )
    num_blocks_n = triton.language.cdiv( size_n, BLOCK_SIZE_N )

    pid_l = pid // ( num_blocks_m * num_blocks_n )
    pid_m = pid % ( num_blocks_m * num_blocks_n )
    pid_m //= num_blocks_n
    pid_n = pid % num_blocks_n

    # calculate offsets for the matrix pointers
    # A offset
    off_a_m = (pid_m*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % size_m
    # B offset
    off_b_n = (pid_n*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % size_n
    # C offset
    off_c_m = pid_m*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_c_n = pid_n*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # L offset
    off_l = pid_l*BLOCK_SIZE_L
    # K offset
    off_k = tl.arange(0, BLOCK_SIZE_K)

    for l in tl.arange(0, BLOCK_SIZE_L):
        # block pointers
        block_ptrs_a  = a_ptr
        if size_a_l > 1:
            block_ptrs_a += (off_l + l) * stride_a_l
        block_ptrs_a += (off_a_m[:, None] * stride_a_m) \
                      + (off_k[None, :]   * stride_a_k)

        block_ptrs_b = b_ptr
        if size_b_l > 1:
            block_ptrs_b += (off_l + l) * stride_b_l
        block_ptrs_b += (off_k[:, None]   * stride_b_k) \
                      + (off_b_n[None, :] * stride_b_n)
        
        # accumulate into [BLOCK_SIZE_M; BLOCK_SIZE_N] Block
        # accumulator
        accum = tl.zeros( ( BLOCK_SIZE_M,
                                        BLOCK_SIZE_N),
                                    dtype = tl.float32 )

        for block_k in range( 0, tl.cdiv( size_k, BLOCK_SIZE_K) ):
            mask_a = off_k[None, :] < ( size_k - block_k * BLOCK_SIZE_K )
            a = tl.load( block_ptrs_a,
                                      mask = mask_a,
                                      other = 0.0 )
            
            mask_b = off_k[:, None] < ( size_k - block_k * BLOCK_SIZE_K )
            b = tl.load( block_ptrs_b,
                                      mask = mask_b,
                                      other = 0.0 )


            accum = tl.dot( a, b, accum, allow_tf32 = False )

            block_ptrs_a += BLOCK_SIZE_K * stride_a_k
            block_ptrs_b += BLOCK_SIZE_K * stride_b_k

        block_ptrs_c = _c_ptr
        block_ptrs_c += (off_l + l) * stride_c_l
        block_ptrs_c += (off_c_m[:, None] * stride_c_m) \
                      + (off_c_n[None, :] * stride_c_n)

        mask_c  = off_c_m[:, None] < size_m
        mask_c &= off_c_n[None, :] < size_n
        tl.store( block_ptrs_c, accum, mask = mask_c )
            

