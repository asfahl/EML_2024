conda activate ai_direct
export QNN_SDK_ROOT=/opt/qcom/aistack/qnn/2.18.0.240101/
source ${QNN_SDK_ROOT}/bin/envsetup.sh
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/x86_64-linux-gnu/:/opt/anaconda3/envs/ai_direct/lib/
export PATH=$PATH:/opt/qcom/HexagonSDK/5.5.0.1/tools/android-ndk-r25c