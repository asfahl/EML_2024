# set serial of android device
export ANDROID_SERIAL=d4334008

# set user directory on the device
export DEVICE_USER_DIR=/data/local/tmp/david/

# assuming that the imagenet data is already present
# assuming a file target_raw_list_host.txt is present on the host

# produces a quantized network which expects 32,224,224,3 input data (see model/resnet18_int8.cpp)
${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-onnx-converter \
  --input_network aimet_export/resnet18/resnet18.onnx \
  --input_list target_raw_list_host.txt \
  --input_encoding 'input_data' other \
  --batch 32 \
  --quantization_overrides aimet_export/resnet18/resnet18.encodings \
  --act_bw=8 \
  --weight_bw=8 \
  --debug \
  --output model/resnet18_int8.cpp

# compile a dynamic library which represents the model
${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c model/resnet18_int8.cpp \
  -b model/resnet18_int8.bin \
  -o model_libs


# generate a serialized context for HTP execution
${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-context-binary-generator \
  --backend ${QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtp.so \
  --model $(pwd)/model_libs/x86_64-linux-clang/libresnet18_int8.so \
  --output_dir model \
  --binary_file resnet18_int8.serialized \
  --log_level info

# create directory to host the model and data on device
adb shell "mkdir -p ${DEVICE_USER_DIR}/resnet18_htp_int8"

# copy the runner and compiled model to the device
adb push ${QNN_SDK_ROOT}/bin/aarch64-android/qnn-net-run ${DEVICE_USER_DIR}/resnet18_htp_int8/
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so ${DEVICE_USER_DIR}/resnet18_htp_int8/
adb push ${QNN_SDK_ROOT}/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so ${DEVICE_USER_DIR}/resnet18_htp_int8/
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV73Stub.so ${DEVICE_USER_DIR}/resnet18_htp_int8/
adb push model/resnet18_int8.serialized.bin ${DEVICE_USER_DIR}/resnet18_htp_int8/

# set up target list on device
adb shell "touch ${DEVICE_USER_DIR}/resnet18_htp_int8/target_raw_list.txt"
for batch in $(seq 0 9); do \
  adb shell "echo ${DEVICE_USER_DIR}/opt/data/imagenet/raw_test/batch_size_32/inputs_${batch}.raw >> ${DEVICE_USER_DIR}/resnet18_htp_int8/target_raw_list.txt"
done

# execute the model on the device HTP
adb shell "cd ${DEVICE_USER_DIR}/resnet18_htp_int8; LD_LIBRARY_PATH=. ./qnn-net-run --backend libQnnHtp.so --retrieve_context resnet18_int8.serialized.bin --input_list target_raw_list.txt"

# copy results from device to host
adb pull ${DEVICE_USER_DIR}/resnet18_htp_int8/output output/htp_int8