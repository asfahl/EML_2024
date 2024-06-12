BUILD_DIR ?= ./build
CXXFLAGS ?=
LDFLAGS ?=
RPATHS ?=
OPTIONS = -O2 -std=c++20 -pedantic -Wall -Wextra -DTORCH_API_INCLUDE_EXTENSION_H -I -D_GLIBCXX_USE_CXX11_ABI=0.

PYTORCH_INCLUDE = $(shell python -c 'from torch.utils.cpp_extension import include_paths; [print(p) for p in include_paths()]')
PYTORCH_LINK = $(shell python -c 'from torch.utils.cpp_extension import library_paths; [print(p) for p in library_paths()]')

CXXFLAGS += $(foreach inc,$(PYTORCH_INCLUDE),-isystem$(inc))
LDFLAGS += $(foreach lin,$(PYTORCH_LINK),-L$(lin)) -lc10 -ltorch -ltorch_cpu -ldl
RPATHS += $(foreach lin,$(PYTORCH_LINK),-Wl,-rpath,$(lin))

$(info $$CXXFLAGS is [${CXXFLAGS}])
$(info $$LDFLAGS is [${LDFLAGS}])

aten: src/aten.cpp
		$(CXX) ${OPTIONS} ${CXXFLAGS} src/aten.cpp -o ${BUILD_DIR}/aten ${RPATHS} ${LDFLAGS}

$(shell mkdir -p ${BUILD_DIR})
