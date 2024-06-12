The file kernels/gemm_asm_asimd_64_64_64.s contains a Neon kernel that performs the operation C+=AB with M=N=K=64 and ldA=ldB=ldC=64.
The kernel is used in the efficient machine learning class at Friedrich Schiller University Jena: https://scalable.uni-jena.de/opt/eml/
The kernel has decent performance, but is not fully tuned.
Expect about 98.5 FP32 GFLOPS on a single NV Grace CPU core.

Usage:

make
./build/gemm_asm_asimd