
# listing 11.1.1
conda activate ai_direct
export QNN_SDK_ROOT=/opt/qcom/aistack/qnn/2.18.0.240101/
source ${QNN_SDK_ROOT}/bin/envsetup.sh
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/x86_64-linux-gnu/:/opt/anaconda3/envs/ai_direct/lib/
export PATH=$PATH:/opt/qcom/HexagonSDK/5.5.0.1/tools/android-ndk-r25c

# listing 11.1.2
# produces a network which expects 32,224,224,3 input data (see model/resnet18_fp32.cpp)
${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-onnx-converter \
  --input_network aimet_export/resnet18/resnet18.onnx \
  --input_encoding 'input' other \
  --batch 32 \
  --debug \
  --output model/resnet18_fp32.cpp

# compile a dynamic library which represents the model
${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c model/resnet18_fp32.cpp \
  -b model/resnet18_fp32.bin \
  -o model_libs

# generate list of inputs
touch target_raw_list_host.txt
for i in $(seq 0 9); do echo /opt/data/imagenet/raw_test/batch_size_32/inputs_$i.raw >> target_raw_list_host.txt; done

# run the model
${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-net-run \
              --log_level=info \
              --backend ${QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnCpu.so \
              --model model_libs/x86_64-linux-clang/libresnet18_fp32.so \
              --input_list target_raw_list_host.txt \
              --output_dir=output/host_fp32