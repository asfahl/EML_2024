BUILD_DIR = ./build

gemm_asm_asimd: driver.cpp kernels/gemm_ref.cpp kernels/gemm_asm_asimd_64_64_64.s
		g++ -g -pedantic -Wall -Wextra -Werror -O2 -c kernels/gemm_ref.cpp -o ${BUILD_DIR}/gemm_ref.o
		as -g kernels/gemm_asm_asimd_64_64_64.s -o ${BUILD_DIR}/gemm_asm_asimd_64_64_64.o
		g++ -g -pedantic -Wall -Wextra -Werror driver.cpp ${BUILD_DIR}/gemm_ref.o ${BUILD_DIR}/gemm_asm_asimd_64_64_64.o -o ${BUILD_DIR}/gemm_asm_asimd

$(shell mkdir -p build)