/* COPYRIGHT HEADER GOES HERE: No CopyRight Header String Passed During Model Conversion */

/* Command Line used:
qnn-onnx-converter act_bitwidth=8 act_bw=8 act_quantizer=tf adjust_nms_features_dims=True algorithms=[] align_matmul_ranks=True arch_checker=False batch=32 bias_bitwidth=8 bias_bw=8 converter_op_package_lib= copyright_file=None custom_io= custom_op_config_paths=None debug=None define_symbol=None disable_batchnorm_folding=False disable_node_validation=False disable_qnn_op_config_validation=False disable_relu_squashing=False dry_run=None dumpIR=False dump_custom_io_config_template= dump_inferred_model=False dump_value_info=False enable_match_gathernd=False exclude_named_tensors=False expand_gru_op_structure=True expand_lstm_op_structure=False extract_color_transform=True float_bias_bitwidth=0 float_bias_bw=32 float_bitwidth=32 float_bw=32 float_fallback=False force_prune_cast_ops=False handle_gather_negative_indices=True ignore_encodings=False inject_cast_for_gather=True input_dim=None input_dtype=[] input_encoding=[['input_data', 'other', 'bgr']] input_layout=[] input_list=target_raw_list_host.txt input_type=[] keep_disconnected_nodes=False keep_int64_inputs=False keep_quant_nodes=False match_caffe_ssd_to_tf=True no_simplification=False op_package_lib= out_names=['class_probs'] overwrite_model_prefix=False pack_4_bit_weights=False package_name=None param_quantizer=None perform_axes_to_spatial_first_order=True prepare_inputs_as_params=False preprocess_roi_pool_inputs=True preserve_io=[] quantization_overrides=aimet_export/resnet18/resnet18.encodings restrict_quantization_steps=[] squash_box_decoder=True unroll_gru_time_steps=True unroll_lstm_time_steps=True use_convert_quantization_nodes=False use_dynamic_16_bit_weights=False use_native_dtype=False use_native_input_files=False use_native_output_files=False use_per_channel_quantization=False use_per_row_quantization=False weight_bw=8 weights_bitwidth=8
*/

#include "QnnOpDef.h"
#include "QnnModel.hpp"

// Flag to determine if Backend should do node validation for each opNode added
#define DO_GRAPH_NODE_VALIDATIONS 1

using namespace qnn_wrapper_api;
extern "C" {
QNN_API
ModelError_t QnnModel_composeGraphs(Qnn_BackendHandle_t backendHandle,
                                    QNN_INTERFACE_VER_TYPE interface,
                                    Qnn_ContextHandle_t contextHandle,
                                    const GraphConfigInfo_t** graphsConfigInfo,
                                    const uint32_t numGraphsConfigInfo,
                                    GraphInfoPtr_t** graphsInfo,
                                    uint32_t* numGraphsInfo,
                                    bool debug,
                                    QnnLog_Callback_t logCallback,
                                    QnnLog_Level_t maxLogLevel) {

  ModelError_t err = MODEL_NO_ERROR;

  /* model/graph for resnet18_int8*/
  QnnModel resnet18_int8;
  const QnnGraph_Config_t** graphConfigs = nullptr;
  VALIDATE(getQnnGraphConfigFromInfo("resnet18_int8", graphsConfigInfo, numGraphsConfigInfo, graphConfigs), err);
  VALIDATE(resnet18_int8.initialize(backendHandle, interface, contextHandle, "resnet18_int8", debug, DO_GRAPH_NODE_VALIDATIONS, graphConfigs), err);
  uint32_t dimensions_input_data[] = {32, 224, 224, 3};
  VALIDATE(resnet18_int8.addTensor("input_data", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "input_data",
                                         .type= QNN_TENSOR_TYPE_APP_WRITE,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0186584480106831f, .offset= -114}}},
                                         .rank= 4,
                                         .dimensions=dimensions_input_data,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=nullptr,
                                                        .dataSize=0}}}}}
  ), err);
  uint32_t dimensions_conv1_weight[] = {7, 7, 3, 64};
  VALIDATE(resnet18_int8.addTensor("conv1_weight", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "conv1_weight",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0080037266016006f, .offset= -128}}},
                                         .rank= 4,
                                         .dimensions=dimensions_conv1_weight,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(conv1_weight),
                                                        .dataSize=BINLEN(conv1_weight)}}}}}
  ), err);
  uint32_t dimensions_conv1_bias[] = {64};
  VALIDATE(resnet18_int8.addTensor("conv1_bias", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "conv1_bias",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0052292402833700f, .offset= -122}}},
                                         .rank= 1,
                                         .dimensions=dimensions_conv1_bias,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(conv1_bias),
                                                        .dataSize=BINLEN(conv1_bias)}}}}}
  ), err);

  /* ADDING NODE FOR conv1 */
  uint32_t dimensions_conv1_dilation[] = {2};
  uint32_t conv1_dilation[] = {1, 1};
  uint32_t dimensions_conv1_pad_amount[] = {2, 2};
  uint32_t conv1_pad_amount[] = {3, 3, 3, 3};
  uint32_t dimensions_conv1_stride[] = {2};
  uint32_t conv1_stride[] = {2, 2};
  Qnn_Param_t params_conv1[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "conv1_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_conv1_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)conv1_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "conv1_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_conv1_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)conv1_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "conv1_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_conv1_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)conv1_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_conv1[] = {
    "input_data",
    "conv1_weight",
    "conv1_bias"
  };
  uint32_t dimensions__130[] = {32, 112, 112, 64};
  Qnn_Tensor_t outputs_conv1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_130",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0243841763585806f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__130,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "conv1", // Node Name
                                 "qti.aisw", // Package Name
                                 "Conv2d", // Qnn Node Type
                                 params_conv1, // Node Params
                                 4, // Num Node Params
                                 inputs_conv1, // Input Tensor Names
                                 3, // Num Input Tensor Names
                                 outputs_conv1, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);


  /* ADDING NODE FOR maxpool */
  uint32_t dimensions_maxpool_filter_size[] = {2};
  uint32_t maxpool_filter_size[] = {3, 3};
  uint32_t dimensions_maxpool_pad_amount[] = {2, 2};
  uint32_t maxpool_pad_amount[] = {1, 0, 1, 0};
  uint32_t dimensions_maxpool_stride[] = {2};
  uint32_t maxpool_stride[] = {2, 2};
  Qnn_Param_t params_maxpool[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="filter_size",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "maxpool_filter_size",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_maxpool_filter_size,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)maxpool_filter_size,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "maxpool_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_maxpool_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)maxpool_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "maxpool_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_maxpool_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)maxpool_stride,
                           .dataSize=8}}}}}}}
  };
  const char*  inputs_maxpool[] = {
    "_130"
  };
  uint32_t dimensions__133[] = {32, 56, 56, 64};
  Qnn_Tensor_t outputs_maxpool[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_133",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0243841763585806f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__133,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "maxpool", // Node Name
                                 "qti.aisw", // Package Name
                                 "PoolMax2d", // Qnn Node Type
                                 params_maxpool, // Node Params
                                 3, // Num Node Params
                                 inputs_maxpool, // Input Tensor Names
                                 1, // Num Input Tensor Names
                                 outputs_maxpool, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);

  uint32_t dimensions_layer1_0_conv1_weight[] = {3, 3, 64, 64};
  VALIDATE(resnet18_int8.addTensor("layer1_0_conv1_weight", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer1_0_conv1_weight",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0062937638722360f, .offset= -128}}},
                                         .rank= 4,
                                         .dimensions=dimensions_layer1_0_conv1_weight,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer1_0_conv1_weight),
                                                        .dataSize=BINLEN(layer1_0_conv1_weight)}}}}}
  ), err);
  uint32_t dimensions_layer1_0_conv1_bias[] = {64};
  VALIDATE(resnet18_int8.addTensor("layer1_0_conv1_bias", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer1_0_conv1_bias",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0075366655364633f, .offset= -109}}},
                                         .rank= 1,
                                         .dimensions=dimensions_layer1_0_conv1_bias,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer1_0_conv1_bias),
                                                        .dataSize=BINLEN(layer1_0_conv1_bias)}}}}}
  ), err);

  /* ADDING NODE FOR layer1_0_conv1 */
  uint32_t dimensions_layer1_0_conv1_dilation[] = {2};
  uint32_t layer1_0_conv1_dilation[] = {1, 1};
  uint32_t dimensions_layer1_0_conv1_pad_amount[] = {2, 2};
  uint32_t layer1_0_conv1_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_layer1_0_conv1_stride[] = {2};
  uint32_t layer1_0_conv1_stride[] = {1, 1};
  Qnn_Param_t params_layer1_0_conv1[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer1_0_conv1_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer1_0_conv1_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer1_0_conv1_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer1_0_conv1_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_layer1_0_conv1_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer1_0_conv1_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer1_0_conv1_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer1_0_conv1_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer1_0_conv1_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_layer1_0_conv1[] = {
    "_133",
    "layer1_0_conv1_weight",
    "layer1_0_conv1_bias"
  };
  uint32_t dimensions__142[] = {32, 56, 56, 64};
  Qnn_Tensor_t outputs_layer1_0_conv1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_142",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0187284220010042f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__142,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "layer1_0_conv1", // Node Name
                                 "qti.aisw", // Package Name
                                 "Conv2d", // Qnn Node Type
                                 params_layer1_0_conv1, // Node Params
                                 4, // Num Node Params
                                 inputs_layer1_0_conv1, // Input Tensor Names
                                 3, // Num Input Tensor Names
                                 outputs_layer1_0_conv1, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);

  uint32_t dimensions_layer1_0_conv2_weight[] = {3, 3, 64, 64};
  VALIDATE(resnet18_int8.addTensor("layer1_0_conv2_weight", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer1_0_conv2_weight",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0038418418262154f, .offset= -128}}},
                                         .rank= 4,
                                         .dimensions=dimensions_layer1_0_conv2_weight,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer1_0_conv2_weight),
                                                        .dataSize=BINLEN(layer1_0_conv2_weight)}}}}}
  ), err);
  uint32_t dimensions_layer1_0_conv2_bias[] = {64};
  VALIDATE(resnet18_int8.addTensor("layer1_0_conv2_bias", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer1_0_conv2_bias",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0119079751893878f, .offset= -105}}},
                                         .rank= 1,
                                         .dimensions=dimensions_layer1_0_conv2_bias,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer1_0_conv2_bias),
                                                        .dataSize=BINLEN(layer1_0_conv2_bias)}}}}}
  ), err);

  /* ADDING NODE FOR layer1_0_conv2 */
  uint32_t dimensions_layer1_0_conv2_dilation[] = {2};
  uint32_t layer1_0_conv2_dilation[] = {1, 1};
  uint32_t dimensions_layer1_0_conv2_pad_amount[] = {2, 2};
  uint32_t layer1_0_conv2_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_layer1_0_conv2_stride[] = {2};
  uint32_t layer1_0_conv2_stride[] = {1, 1};
  Qnn_Param_t params_layer1_0_conv2[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer1_0_conv2_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer1_0_conv2_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer1_0_conv2_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer1_0_conv2_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_layer1_0_conv2_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer1_0_conv2_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer1_0_conv2_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer1_0_conv2_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer1_0_conv2_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_layer1_0_conv2[] = {
    "_142",
    "layer1_0_conv2_weight",
    "layer1_0_conv2_bias"
  };
  uint32_t dimensions__148[] = {32, 56, 56, 64};
  Qnn_Tensor_t outputs_layer1_0_conv2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_148",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0496850386261940f, .offset= -133}}},
            .rank= 4,
            .dimensions=dimensions__148,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "layer1_0_conv2", // Node Name
                                 "qti.aisw", // Package Name
                                 "Conv2d", // Qnn Node Type
                                 params_layer1_0_conv2, // Node Params
                                 4, // Num Node Params
                                 inputs_layer1_0_conv2, // Input Tensor Names
                                 3, // Num Input Tensor Names
                                 outputs_layer1_0_conv2, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);


  /* ADDING NODE FOR module_add */
  const char*  inputs_module_add[] = {
    "_148",
    "_133"
  };
  uint32_t dimensions__155[] = {32, 56, 56, 64};
  Qnn_Tensor_t outputs_module_add[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_155",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0267292615026236f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__155,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "module_add", // Node Name
                                 "qti.aisw", // Package Name
                                 "ElementWiseAdd", // Qnn Node Type
                                 nullptr, // Node Params
                                 0, // Num Node Params
                                 inputs_module_add, // Input Tensor Names
                                 2, // Num Input Tensor Names
                                 outputs_module_add, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);

  uint32_t dimensions_layer1_1_conv1_weight[] = {3, 3, 64, 64};
  VALIDATE(resnet18_int8.addTensor("layer1_1_conv1_weight", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer1_1_conv1_weight",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0051111695356667f, .offset= -128}}},
                                         .rank= 4,
                                         .dimensions=dimensions_layer1_1_conv1_weight,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer1_1_conv1_weight),
                                                        .dataSize=BINLEN(layer1_1_conv1_weight)}}}}}
  ), err);
  uint32_t dimensions_layer1_1_conv1_bias[] = {64};
  VALIDATE(resnet18_int8.addTensor("layer1_1_conv1_bias", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer1_1_conv1_bias",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0087309069931507f, .offset= -117}}},
                                         .rank= 1,
                                         .dimensions=dimensions_layer1_1_conv1_bias,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer1_1_conv1_bias),
                                                        .dataSize=BINLEN(layer1_1_conv1_bias)}}}}}
  ), err);

  /* ADDING NODE FOR layer1_1_conv1 */
  uint32_t dimensions_layer1_1_conv1_dilation[] = {2};
  uint32_t layer1_1_conv1_dilation[] = {1, 1};
  uint32_t dimensions_layer1_1_conv1_pad_amount[] = {2, 2};
  uint32_t layer1_1_conv1_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_layer1_1_conv1_stride[] = {2};
  uint32_t layer1_1_conv1_stride[] = {1, 1};
  Qnn_Param_t params_layer1_1_conv1[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer1_1_conv1_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer1_1_conv1_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer1_1_conv1_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer1_1_conv1_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_layer1_1_conv1_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer1_1_conv1_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer1_1_conv1_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer1_1_conv1_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer1_1_conv1_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_layer1_1_conv1[] = {
    "_155",
    "layer1_1_conv1_weight",
    "layer1_1_conv1_bias"
  };
  uint32_t dimensions__164[] = {32, 56, 56, 64};
  Qnn_Tensor_t outputs_layer1_1_conv1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_164",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0164157822728157f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__164,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "layer1_1_conv1", // Node Name
                                 "qti.aisw", // Package Name
                                 "Conv2d", // Qnn Node Type
                                 params_layer1_1_conv1, // Node Params
                                 4, // Num Node Params
                                 inputs_layer1_1_conv1, // Input Tensor Names
                                 3, // Num Input Tensor Names
                                 outputs_layer1_1_conv1, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);

  uint32_t dimensions_layer1_1_conv2_weight[] = {3, 3, 64, 64};
  VALIDATE(resnet18_int8.addTensor("layer1_1_conv2_weight", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer1_1_conv2_weight",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0029969410970807f, .offset= -128}}},
                                         .rank= 4,
                                         .dimensions=dimensions_layer1_1_conv2_weight,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer1_1_conv2_weight),
                                                        .dataSize=BINLEN(layer1_1_conv2_weight)}}}}}
  ), err);
  uint32_t dimensions_layer1_1_conv2_bias[] = {64};
  VALIDATE(resnet18_int8.addTensor("layer1_1_conv2_bias", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer1_1_conv2_bias",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0088430047035217f, .offset= -132}}},
                                         .rank= 1,
                                         .dimensions=dimensions_layer1_1_conv2_bias,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer1_1_conv2_bias),
                                                        .dataSize=BINLEN(layer1_1_conv2_bias)}}}}}
  ), err);

  /* ADDING NODE FOR layer1_1_conv2 */
  uint32_t dimensions_layer1_1_conv2_dilation[] = {2};
  uint32_t layer1_1_conv2_dilation[] = {1, 1};
  uint32_t dimensions_layer1_1_conv2_pad_amount[] = {2, 2};
  uint32_t layer1_1_conv2_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_layer1_1_conv2_stride[] = {2};
  uint32_t layer1_1_conv2_stride[] = {1, 1};
  Qnn_Param_t params_layer1_1_conv2[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer1_1_conv2_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer1_1_conv2_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer1_1_conv2_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer1_1_conv2_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_layer1_1_conv2_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer1_1_conv2_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer1_1_conv2_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer1_1_conv2_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer1_1_conv2_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_layer1_1_conv2[] = {
    "_164",
    "layer1_1_conv2_weight",
    "layer1_1_conv2_bias"
  };
  uint32_t dimensions__170[] = {32, 56, 56, 64};
  Qnn_Tensor_t outputs_layer1_1_conv2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_170",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0739980340003967f, .offset= -164}}},
            .rank= 4,
            .dimensions=dimensions__170,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "layer1_1_conv2", // Node Name
                                 "qti.aisw", // Package Name
                                 "Conv2d", // Qnn Node Type
                                 params_layer1_1_conv2, // Node Params
                                 4, // Num Node Params
                                 inputs_layer1_1_conv2, // Input Tensor Names
                                 3, // Num Input Tensor Names
                                 outputs_layer1_1_conv2, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);


  /* ADDING NODE FOR module_add_1 */
  const char*  inputs_module_add_1[] = {
    "_170",
    "_155"
  };
  uint32_t dimensions__177[] = {32, 56, 56, 64};
  Qnn_Tensor_t outputs_module_add_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_177",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0338630899786949f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__177,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "module_add_1", // Node Name
                                 "qti.aisw", // Package Name
                                 "ElementWiseAdd", // Qnn Node Type
                                 nullptr, // Node Params
                                 0, // Num Node Params
                                 inputs_module_add_1, // Input Tensor Names
                                 2, // Num Input Tensor Names
                                 outputs_module_add_1, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);

  uint32_t dimensions_layer2_0_conv1_weight[] = {3, 3, 64, 128};
  VALIDATE(resnet18_int8.addTensor("layer2_0_conv1_weight", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer2_0_conv1_weight",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0026814034208655f, .offset= -128}}},
                                         .rank= 4,
                                         .dimensions=dimensions_layer2_0_conv1_weight,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer2_0_conv1_weight),
                                                        .dataSize=BINLEN(layer2_0_conv1_weight)}}}}}
  ), err);
  uint32_t dimensions_layer2_0_conv1_bias[] = {128};
  VALIDATE(resnet18_int8.addTensor("layer2_0_conv1_bias", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer2_0_conv1_bias",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0046267709694803f, .offset= -95}}},
                                         .rank= 1,
                                         .dimensions=dimensions_layer2_0_conv1_bias,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer2_0_conv1_bias),
                                                        .dataSize=BINLEN(layer2_0_conv1_bias)}}}}}
  ), err);

  /* ADDING NODE FOR layer2_0_conv1 */
  uint32_t dimensions_layer2_0_conv1_dilation[] = {2};
  uint32_t layer2_0_conv1_dilation[] = {1, 1};
  uint32_t dimensions_layer2_0_conv1_pad_amount[] = {2, 2};
  uint32_t layer2_0_conv1_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_layer2_0_conv1_stride[] = {2};
  uint32_t layer2_0_conv1_stride[] = {2, 2};
  Qnn_Param_t params_layer2_0_conv1[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer2_0_conv1_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer2_0_conv1_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer2_0_conv1_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer2_0_conv1_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_layer2_0_conv1_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer2_0_conv1_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer2_0_conv1_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer2_0_conv1_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer2_0_conv1_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_layer2_0_conv1[] = {
    "_177",
    "layer2_0_conv1_weight",
    "layer2_0_conv1_bias"
  };
  uint32_t dimensions__186[] = {32, 28, 28, 128};
  Qnn_Tensor_t outputs_layer2_0_conv1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_186",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0149837862700224f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__186,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "layer2_0_conv1", // Node Name
                                 "qti.aisw", // Package Name
                                 "Conv2d", // Qnn Node Type
                                 params_layer2_0_conv1, // Node Params
                                 4, // Num Node Params
                                 inputs_layer2_0_conv1, // Input Tensor Names
                                 3, // Num Input Tensor Names
                                 outputs_layer2_0_conv1, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);

  uint32_t dimensions_layer2_0_conv2_weight[] = {3, 3, 128, 128};
  VALIDATE(resnet18_int8.addTensor("layer2_0_conv2_weight", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer2_0_conv2_weight",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0033634640276432f, .offset= -128}}},
                                         .rank= 4,
                                         .dimensions=dimensions_layer2_0_conv2_weight,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer2_0_conv2_weight),
                                                        .dataSize=BINLEN(layer2_0_conv2_weight)}}}}}
  ), err);
  uint32_t dimensions_layer2_0_conv2_bias[] = {128};
  VALIDATE(resnet18_int8.addTensor("layer2_0_conv2_bias", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer2_0_conv2_bias",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0085827028378844f, .offset= -86}}},
                                         .rank= 1,
                                         .dimensions=dimensions_layer2_0_conv2_bias,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer2_0_conv2_bias),
                                                        .dataSize=BINLEN(layer2_0_conv2_bias)}}}}}
  ), err);

  /* ADDING NODE FOR layer2_0_conv2 */
  uint32_t dimensions_layer2_0_conv2_dilation[] = {2};
  uint32_t layer2_0_conv2_dilation[] = {1, 1};
  uint32_t dimensions_layer2_0_conv2_pad_amount[] = {2, 2};
  uint32_t layer2_0_conv2_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_layer2_0_conv2_stride[] = {2};
  uint32_t layer2_0_conv2_stride[] = {1, 1};
  Qnn_Param_t params_layer2_0_conv2[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer2_0_conv2_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer2_0_conv2_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer2_0_conv2_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer2_0_conv2_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_layer2_0_conv2_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer2_0_conv2_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer2_0_conv2_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer2_0_conv2_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer2_0_conv2_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_layer2_0_conv2[] = {
    "_186",
    "layer2_0_conv2_weight",
    "layer2_0_conv2_bias"
  };
  uint32_t dimensions__192[] = {32, 28, 28, 128};
  Qnn_Tensor_t outputs_layer2_0_conv2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_192",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0431009232997894f, .offset= -102}}},
            .rank= 4,
            .dimensions=dimensions__192,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "layer2_0_conv2", // Node Name
                                 "qti.aisw", // Package Name
                                 "Conv2d", // Qnn Node Type
                                 params_layer2_0_conv2, // Node Params
                                 4, // Num Node Params
                                 inputs_layer2_0_conv2, // Input Tensor Names
                                 3, // Num Input Tensor Names
                                 outputs_layer2_0_conv2, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);

  uint32_t dimensions_layer2_0_downsample_0_weight[] = {1, 1, 64, 128};
  VALIDATE(resnet18_int8.addTensor("layer2_0_downsample_0_weight", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer2_0_downsample_0_weight",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0061435215175152f, .offset= -128}}},
                                         .rank= 4,
                                         .dimensions=dimensions_layer2_0_downsample_0_weight,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer2_0_downsample_0_weight),
                                                        .dataSize=BINLEN(layer2_0_downsample_0_weight)}}}}}
  ), err);
  uint32_t dimensions_layer2_0_downsample_0_bias[] = {128};
  VALIDATE(resnet18_int8.addTensor("layer2_0_downsample_0_bias", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer2_0_downsample_0_bias",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0081412447616458f, .offset= -136}}},
                                         .rank= 1,
                                         .dimensions=dimensions_layer2_0_downsample_0_bias,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer2_0_downsample_0_bias),
                                                        .dataSize=BINLEN(layer2_0_downsample_0_bias)}}}}}
  ), err);

  /* ADDING NODE FOR layer2_0_downsample_0 */
  uint32_t dimensions_layer2_0_downsample_0_dilation[] = {2};
  uint32_t layer2_0_downsample_0_dilation[] = {1, 1};
  uint32_t dimensions_layer2_0_downsample_0_pad_amount[] = {2, 2};
  uint32_t layer2_0_downsample_0_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_layer2_0_downsample_0_stride[] = {2};
  uint32_t layer2_0_downsample_0_stride[] = {2, 2};
  Qnn_Param_t params_layer2_0_downsample_0[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer2_0_downsample_0_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer2_0_downsample_0_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer2_0_downsample_0_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer2_0_downsample_0_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_layer2_0_downsample_0_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer2_0_downsample_0_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer2_0_downsample_0_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer2_0_downsample_0_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer2_0_downsample_0_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_layer2_0_downsample_0[] = {
    "_177",
    "layer2_0_downsample_0_weight",
    "layer2_0_downsample_0_bias"
  };
  uint32_t dimensions__198[] = {32, 28, 28, 128};
  Qnn_Tensor_t outputs_layer2_0_downsample_0[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_198",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0377241298556328f, .offset= -141}}},
            .rank= 4,
            .dimensions=dimensions__198,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "layer2_0_downsample_0", // Node Name
                                 "qti.aisw", // Package Name
                                 "Conv2d", // Qnn Node Type
                                 params_layer2_0_downsample_0, // Node Params
                                 4, // Num Node Params
                                 inputs_layer2_0_downsample_0, // Input Tensor Names
                                 3, // Num Input Tensor Names
                                 outputs_layer2_0_downsample_0, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);


  /* ADDING NODE FOR module_add_2 */
  const char*  inputs_module_add_2[] = {
    "_192",
    "_198"
  };
  uint32_t dimensions__205[] = {32, 28, 28, 128};
  Qnn_Tensor_t outputs_module_add_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_205",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0291704013943672f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__205,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "module_add_2", // Node Name
                                 "qti.aisw", // Package Name
                                 "ElementWiseAdd", // Qnn Node Type
                                 nullptr, // Node Params
                                 0, // Num Node Params
                                 inputs_module_add_2, // Input Tensor Names
                                 2, // Num Input Tensor Names
                                 outputs_module_add_2, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);

  uint32_t dimensions_layer2_1_conv1_weight[] = {3, 3, 128, 128};
  VALIDATE(resnet18_int8.addTensor("layer2_1_conv1_weight", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer2_1_conv1_weight",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0034547105897218f, .offset= -128}}},
                                         .rank= 4,
                                         .dimensions=dimensions_layer2_1_conv1_weight,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer2_1_conv1_weight),
                                                        .dataSize=BINLEN(layer2_1_conv1_weight)}}}}}
  ), err);
  uint32_t dimensions_layer2_1_conv1_bias[] = {128};
  VALIDATE(resnet18_int8.addTensor("layer2_1_conv1_bias", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer2_1_conv1_bias",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0056576351635158f, .offset= -146}}},
                                         .rank= 1,
                                         .dimensions=dimensions_layer2_1_conv1_bias,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer2_1_conv1_bias),
                                                        .dataSize=BINLEN(layer2_1_conv1_bias)}}}}}
  ), err);

  /* ADDING NODE FOR layer2_1_conv1 */
  uint32_t dimensions_layer2_1_conv1_dilation[] = {2};
  uint32_t layer2_1_conv1_dilation[] = {1, 1};
  uint32_t dimensions_layer2_1_conv1_pad_amount[] = {2, 2};
  uint32_t layer2_1_conv1_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_layer2_1_conv1_stride[] = {2};
  uint32_t layer2_1_conv1_stride[] = {1, 1};
  Qnn_Param_t params_layer2_1_conv1[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer2_1_conv1_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer2_1_conv1_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer2_1_conv1_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer2_1_conv1_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_layer2_1_conv1_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer2_1_conv1_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer2_1_conv1_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer2_1_conv1_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer2_1_conv1_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_layer2_1_conv1[] = {
    "_205",
    "layer2_1_conv1_weight",
    "layer2_1_conv1_bias"
  };
  uint32_t dimensions__214[] = {32, 28, 28, 128};
  Qnn_Tensor_t outputs_layer2_1_conv1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_214",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0194325111806393f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__214,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "layer2_1_conv1", // Node Name
                                 "qti.aisw", // Package Name
                                 "Conv2d", // Qnn Node Type
                                 params_layer2_1_conv1, // Node Params
                                 4, // Num Node Params
                                 inputs_layer2_1_conv1, // Input Tensor Names
                                 3, // Num Input Tensor Names
                                 outputs_layer2_1_conv1, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);

  uint32_t dimensions_layer2_1_conv2_weight[] = {3, 3, 128, 128};
  VALIDATE(resnet18_int8.addTensor("layer2_1_conv2_weight", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer2_1_conv2_weight",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0028004534542561f, .offset= -128}}},
                                         .rank= 4,
                                         .dimensions=dimensions_layer2_1_conv2_weight,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer2_1_conv2_weight),
                                                        .dataSize=BINLEN(layer2_1_conv2_weight)}}}}}
  ), err);
  uint32_t dimensions_layer2_1_conv2_bias[] = {128};
  VALIDATE(resnet18_int8.addTensor("layer2_1_conv2_bias", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer2_1_conv2_bias",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0090296454727650f, .offset= -127}}},
                                         .rank= 1,
                                         .dimensions=dimensions_layer2_1_conv2_bias,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer2_1_conv2_bias),
                                                        .dataSize=BINLEN(layer2_1_conv2_bias)}}}}}
  ), err);

  /* ADDING NODE FOR layer2_1_conv2 */
  uint32_t dimensions_layer2_1_conv2_dilation[] = {2};
  uint32_t layer2_1_conv2_dilation[] = {1, 1};
  uint32_t dimensions_layer2_1_conv2_pad_amount[] = {2, 2};
  uint32_t layer2_1_conv2_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_layer2_1_conv2_stride[] = {2};
  uint32_t layer2_1_conv2_stride[] = {1, 1};
  Qnn_Param_t params_layer2_1_conv2[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer2_1_conv2_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer2_1_conv2_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer2_1_conv2_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer2_1_conv2_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_layer2_1_conv2_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer2_1_conv2_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer2_1_conv2_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer2_1_conv2_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer2_1_conv2_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_layer2_1_conv2[] = {
    "_214",
    "layer2_1_conv2_weight",
    "layer2_1_conv2_bias"
  };
  uint32_t dimensions__220[] = {32, 28, 28, 128};
  Qnn_Tensor_t outputs_layer2_1_conv2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_220",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0494959391653538f, .offset= -157}}},
            .rank= 4,
            .dimensions=dimensions__220,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "layer2_1_conv2", // Node Name
                                 "qti.aisw", // Package Name
                                 "Conv2d", // Qnn Node Type
                                 params_layer2_1_conv2, // Node Params
                                 4, // Num Node Params
                                 inputs_layer2_1_conv2, // Input Tensor Names
                                 3, // Num Input Tensor Names
                                 outputs_layer2_1_conv2, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);


  /* ADDING NODE FOR module_add_3 */
  const char*  inputs_module_add_3[] = {
    "_220",
    "_205"
  };
  uint32_t dimensions__227[] = {32, 28, 28, 128};
  Qnn_Tensor_t outputs_module_add_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_227",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0291318222880363f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__227,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "module_add_3", // Node Name
                                 "qti.aisw", // Package Name
                                 "ElementWiseAdd", // Qnn Node Type
                                 nullptr, // Node Params
                                 0, // Num Node Params
                                 inputs_module_add_3, // Input Tensor Names
                                 2, // Num Input Tensor Names
                                 outputs_module_add_3, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);

  uint32_t dimensions_layer3_0_conv1_weight[] = {3, 3, 128, 256};
  VALIDATE(resnet18_int8.addTensor("layer3_0_conv1_weight", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer3_0_conv1_weight",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0030862218700349f, .offset= -128}}},
                                         .rank= 4,
                                         .dimensions=dimensions_layer3_0_conv1_weight,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer3_0_conv1_weight),
                                                        .dataSize=BINLEN(layer3_0_conv1_weight)}}}}}
  ), err);
  uint32_t dimensions_layer3_0_conv1_bias[] = {256};
  VALIDATE(resnet18_int8.addTensor("layer3_0_conv1_bias", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer3_0_conv1_bias",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0060660871677101f, .offset= -113}}},
                                         .rank= 1,
                                         .dimensions=dimensions_layer3_0_conv1_bias,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer3_0_conv1_bias),
                                                        .dataSize=BINLEN(layer3_0_conv1_bias)}}}}}
  ), err);

  /* ADDING NODE FOR layer3_0_conv1 */
  uint32_t dimensions_layer3_0_conv1_dilation[] = {2};
  uint32_t layer3_0_conv1_dilation[] = {1, 1};
  uint32_t dimensions_layer3_0_conv1_pad_amount[] = {2, 2};
  uint32_t layer3_0_conv1_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_layer3_0_conv1_stride[] = {2};
  uint32_t layer3_0_conv1_stride[] = {2, 2};
  Qnn_Param_t params_layer3_0_conv1[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer3_0_conv1_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer3_0_conv1_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer3_0_conv1_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer3_0_conv1_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_layer3_0_conv1_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer3_0_conv1_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer3_0_conv1_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer3_0_conv1_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer3_0_conv1_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_layer3_0_conv1[] = {
    "_227",
    "layer3_0_conv1_weight",
    "layer3_0_conv1_bias"
  };
  uint32_t dimensions__236[] = {32, 14, 14, 256};
  Qnn_Tensor_t outputs_layer3_0_conv1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_236",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0167406518012285f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__236,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "layer3_0_conv1", // Node Name
                                 "qti.aisw", // Package Name
                                 "Conv2d", // Qnn Node Type
                                 params_layer3_0_conv1, // Node Params
                                 4, // Num Node Params
                                 inputs_layer3_0_conv1, // Input Tensor Names
                                 3, // Num Input Tensor Names
                                 outputs_layer3_0_conv1, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);

  uint32_t dimensions_layer3_0_conv2_weight[] = {3, 3, 256, 256};
  VALIDATE(resnet18_int8.addTensor("layer3_0_conv2_weight", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer3_0_conv2_weight",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0026260958984494f, .offset= -128}}},
                                         .rank= 4,
                                         .dimensions=dimensions_layer3_0_conv2_weight,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer3_0_conv2_weight),
                                                        .dataSize=BINLEN(layer3_0_conv2_weight)}}}}}
  ), err);
  uint32_t dimensions_layer3_0_conv2_bias[] = {256};
  VALIDATE(resnet18_int8.addTensor("layer3_0_conv2_bias", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer3_0_conv2_bias",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0038371533155441f, .offset= -96}}},
                                         .rank= 1,
                                         .dimensions=dimensions_layer3_0_conv2_bias,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer3_0_conv2_bias),
                                                        .dataSize=BINLEN(layer3_0_conv2_bias)}}}}}
  ), err);

  /* ADDING NODE FOR layer3_0_conv2 */
  uint32_t dimensions_layer3_0_conv2_dilation[] = {2};
  uint32_t layer3_0_conv2_dilation[] = {1, 1};
  uint32_t dimensions_layer3_0_conv2_pad_amount[] = {2, 2};
  uint32_t layer3_0_conv2_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_layer3_0_conv2_stride[] = {2};
  uint32_t layer3_0_conv2_stride[] = {1, 1};
  Qnn_Param_t params_layer3_0_conv2[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer3_0_conv2_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer3_0_conv2_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer3_0_conv2_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer3_0_conv2_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_layer3_0_conv2_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer3_0_conv2_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer3_0_conv2_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer3_0_conv2_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer3_0_conv2_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_layer3_0_conv2[] = {
    "_236",
    "layer3_0_conv2_weight",
    "layer3_0_conv2_bias"
  };
  uint32_t dimensions__242[] = {32, 14, 14, 256};
  Qnn_Tensor_t outputs_layer3_0_conv2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_242",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0378371514379978f, .offset= -92}}},
            .rank= 4,
            .dimensions=dimensions__242,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "layer3_0_conv2", // Node Name
                                 "qti.aisw", // Package Name
                                 "Conv2d", // Qnn Node Type
                                 params_layer3_0_conv2, // Node Params
                                 4, // Num Node Params
                                 inputs_layer3_0_conv2, // Input Tensor Names
                                 3, // Num Input Tensor Names
                                 outputs_layer3_0_conv2, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);

  uint32_t dimensions_layer3_0_downsample_0_weight[] = {1, 1, 128, 256};
  VALIDATE(resnet18_int8.addTensor("layer3_0_downsample_0_weight", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer3_0_downsample_0_weight",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0021074807737023f, .offset= -128}}},
                                         .rank= 4,
                                         .dimensions=dimensions_layer3_0_downsample_0_weight,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer3_0_downsample_0_weight),
                                                        .dataSize=BINLEN(layer3_0_downsample_0_weight)}}}}}
  ), err);
  uint32_t dimensions_layer3_0_downsample_0_bias[] = {256};
  VALIDATE(resnet18_int8.addTensor("layer3_0_downsample_0_bias", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer3_0_downsample_0_bias",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0022548795677722f, .offset= -149}}},
                                         .rank= 1,
                                         .dimensions=dimensions_layer3_0_downsample_0_bias,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer3_0_downsample_0_bias),
                                                        .dataSize=BINLEN(layer3_0_downsample_0_bias)}}}}}
  ), err);

  /* ADDING NODE FOR layer3_0_downsample_0 */
  uint32_t dimensions_layer3_0_downsample_0_dilation[] = {2};
  uint32_t layer3_0_downsample_0_dilation[] = {1, 1};
  uint32_t dimensions_layer3_0_downsample_0_pad_amount[] = {2, 2};
  uint32_t layer3_0_downsample_0_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_layer3_0_downsample_0_stride[] = {2};
  uint32_t layer3_0_downsample_0_stride[] = {2, 2};
  Qnn_Param_t params_layer3_0_downsample_0[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer3_0_downsample_0_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer3_0_downsample_0_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer3_0_downsample_0_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer3_0_downsample_0_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_layer3_0_downsample_0_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer3_0_downsample_0_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer3_0_downsample_0_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer3_0_downsample_0_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer3_0_downsample_0_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_layer3_0_downsample_0[] = {
    "_227",
    "layer3_0_downsample_0_weight",
    "layer3_0_downsample_0_bias"
  };
  uint32_t dimensions__248[] = {32, 14, 14, 256};
  Qnn_Tensor_t outputs_layer3_0_downsample_0[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_248",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0169508196413517f, .offset= -156}}},
            .rank= 4,
            .dimensions=dimensions__248,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "layer3_0_downsample_0", // Node Name
                                 "qti.aisw", // Package Name
                                 "Conv2d", // Qnn Node Type
                                 params_layer3_0_downsample_0, // Node Params
                                 4, // Num Node Params
                                 inputs_layer3_0_downsample_0, // Input Tensor Names
                                 3, // Num Input Tensor Names
                                 outputs_layer3_0_downsample_0, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);


  /* ADDING NODE FOR module_add_4 */
  const char*  inputs_module_add_4[] = {
    "_242",
    "_248"
  };
  uint32_t dimensions__255[] = {32, 14, 14, 256};
  Qnn_Tensor_t outputs_module_add_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_255",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0246487576514482f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__255,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "module_add_4", // Node Name
                                 "qti.aisw", // Package Name
                                 "ElementWiseAdd", // Qnn Node Type
                                 nullptr, // Node Params
                                 0, // Num Node Params
                                 inputs_module_add_4, // Input Tensor Names
                                 2, // Num Input Tensor Names
                                 outputs_module_add_4, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);

  uint32_t dimensions_layer3_1_conv1_weight[] = {3, 3, 256, 256};
  VALIDATE(resnet18_int8.addTensor("layer3_1_conv1_weight", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer3_1_conv1_weight",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0023310356773436f, .offset= -128}}},
                                         .rank= 4,
                                         .dimensions=dimensions_layer3_1_conv1_weight,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer3_1_conv1_weight),
                                                        .dataSize=BINLEN(layer3_1_conv1_weight)}}}}}
  ), err);
  uint32_t dimensions_layer3_1_conv1_bias[] = {256};
  VALIDATE(resnet18_int8.addTensor("layer3_1_conv1_bias", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer3_1_conv1_bias",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0060801822692156f, .offset= -133}}},
                                         .rank= 1,
                                         .dimensions=dimensions_layer3_1_conv1_bias,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer3_1_conv1_bias),
                                                        .dataSize=BINLEN(layer3_1_conv1_bias)}}}}}
  ), err);

  /* ADDING NODE FOR layer3_1_conv1 */
  uint32_t dimensions_layer3_1_conv1_dilation[] = {2};
  uint32_t layer3_1_conv1_dilation[] = {1, 1};
  uint32_t dimensions_layer3_1_conv1_pad_amount[] = {2, 2};
  uint32_t layer3_1_conv1_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_layer3_1_conv1_stride[] = {2};
  uint32_t layer3_1_conv1_stride[] = {1, 1};
  Qnn_Param_t params_layer3_1_conv1[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer3_1_conv1_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer3_1_conv1_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer3_1_conv1_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer3_1_conv1_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_layer3_1_conv1_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer3_1_conv1_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer3_1_conv1_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer3_1_conv1_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer3_1_conv1_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_layer3_1_conv1[] = {
    "_255",
    "layer3_1_conv1_weight",
    "layer3_1_conv1_bias"
  };
  uint32_t dimensions__264[] = {32, 14, 14, 256};
  Qnn_Tensor_t outputs_layer3_1_conv1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_264",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0195951256901026f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__264,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "layer3_1_conv1", // Node Name
                                 "qti.aisw", // Package Name
                                 "Conv2d", // Qnn Node Type
                                 params_layer3_1_conv1, // Node Params
                                 4, // Num Node Params
                                 inputs_layer3_1_conv1, // Input Tensor Names
                                 3, // Num Input Tensor Names
                                 outputs_layer3_1_conv1, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);

  uint32_t dimensions_layer3_1_conv2_weight[] = {3, 3, 256, 256};
  VALIDATE(resnet18_int8.addTensor("layer3_1_conv2_weight", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer3_1_conv2_weight",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0025948574766517f, .offset= -128}}},
                                         .rank= 4,
                                         .dimensions=dimensions_layer3_1_conv2_weight,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer3_1_conv2_weight),
                                                        .dataSize=BINLEN(layer3_1_conv2_weight)}}}}}
  ), err);
  uint32_t dimensions_layer3_1_conv2_bias[] = {256};
  VALIDATE(resnet18_int8.addTensor("layer3_1_conv2_bias", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer3_1_conv2_bias",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0081342179328203f, .offset= -125}}},
                                         .rank= 1,
                                         .dimensions=dimensions_layer3_1_conv2_bias,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer3_1_conv2_bias),
                                                        .dataSize=BINLEN(layer3_1_conv2_bias)}}}}}
  ), err);

  /* ADDING NODE FOR layer3_1_conv2 */
  uint32_t dimensions_layer3_1_conv2_dilation[] = {2};
  uint32_t layer3_1_conv2_dilation[] = {1, 1};
  uint32_t dimensions_layer3_1_conv2_pad_amount[] = {2, 2};
  uint32_t layer3_1_conv2_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_layer3_1_conv2_stride[] = {2};
  uint32_t layer3_1_conv2_stride[] = {1, 1};
  Qnn_Param_t params_layer3_1_conv2[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer3_1_conv2_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer3_1_conv2_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer3_1_conv2_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer3_1_conv2_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_layer3_1_conv2_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer3_1_conv2_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer3_1_conv2_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer3_1_conv2_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer3_1_conv2_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_layer3_1_conv2[] = {
    "_264",
    "layer3_1_conv2_weight",
    "layer3_1_conv2_bias"
  };
  uint32_t dimensions__270[] = {32, 14, 14, 256};
  Qnn_Tensor_t outputs_layer3_1_conv2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_270",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0494119115173817f, .offset= -159}}},
            .rank= 4,
            .dimensions=dimensions__270,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "layer3_1_conv2", // Node Name
                                 "qti.aisw", // Package Name
                                 "Conv2d", // Qnn Node Type
                                 params_layer3_1_conv2, // Node Params
                                 4, // Num Node Params
                                 inputs_layer3_1_conv2, // Input Tensor Names
                                 3, // Num Input Tensor Names
                                 outputs_layer3_1_conv2, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);


  /* ADDING NODE FOR module_add_5 */
  const char*  inputs_module_add_5[] = {
    "_270",
    "_255"
  };
  uint32_t dimensions__277[] = {32, 14, 14, 256};
  Qnn_Tensor_t outputs_module_add_5[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_277",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0356519520282745f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__277,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "module_add_5", // Node Name
                                 "qti.aisw", // Package Name
                                 "ElementWiseAdd", // Qnn Node Type
                                 nullptr, // Node Params
                                 0, // Num Node Params
                                 inputs_module_add_5, // Input Tensor Names
                                 2, // Num Input Tensor Names
                                 outputs_module_add_5, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);

  uint32_t dimensions_layer4_0_conv1_weight[] = {3, 3, 256, 512};
  VALIDATE(resnet18_int8.addTensor("layer4_0_conv1_weight", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer4_0_conv1_weight",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0030127656646073f, .offset= -128}}},
                                         .rank= 4,
                                         .dimensions=dimensions_layer4_0_conv1_weight,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer4_0_conv1_weight),
                                                        .dataSize=BINLEN(layer4_0_conv1_weight)}}}}}
  ), err);
  uint32_t dimensions_layer4_0_conv1_bias[] = {512};
  VALIDATE(resnet18_int8.addTensor("layer4_0_conv1_bias", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer4_0_conv1_bias",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0040908488444984f, .offset= -117}}},
                                         .rank= 1,
                                         .dimensions=dimensions_layer4_0_conv1_bias,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer4_0_conv1_bias),
                                                        .dataSize=BINLEN(layer4_0_conv1_bias)}}}}}
  ), err);

  /* ADDING NODE FOR layer4_0_conv1 */
  uint32_t dimensions_layer4_0_conv1_dilation[] = {2};
  uint32_t layer4_0_conv1_dilation[] = {1, 1};
  uint32_t dimensions_layer4_0_conv1_pad_amount[] = {2, 2};
  uint32_t layer4_0_conv1_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_layer4_0_conv1_stride[] = {2};
  uint32_t layer4_0_conv1_stride[] = {2, 2};
  Qnn_Param_t params_layer4_0_conv1[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer4_0_conv1_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer4_0_conv1_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer4_0_conv1_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer4_0_conv1_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_layer4_0_conv1_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer4_0_conv1_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer4_0_conv1_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer4_0_conv1_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer4_0_conv1_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_layer4_0_conv1[] = {
    "_277",
    "layer4_0_conv1_weight",
    "layer4_0_conv1_bias"
  };
  uint32_t dimensions__286[] = {32, 7, 7, 512};
  Qnn_Tensor_t outputs_layer4_0_conv1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_286",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0114180631935596f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__286,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "layer4_0_conv1", // Node Name
                                 "qti.aisw", // Package Name
                                 "Conv2d", // Qnn Node Type
                                 params_layer4_0_conv1, // Node Params
                                 4, // Num Node Params
                                 inputs_layer4_0_conv1, // Input Tensor Names
                                 3, // Num Input Tensor Names
                                 outputs_layer4_0_conv1, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);

  uint32_t dimensions_layer4_0_conv2_weight[] = {3, 3, 512, 512};
  VALIDATE(resnet18_int8.addTensor("layer4_0_conv2_weight", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer4_0_conv2_weight",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0027458304539323f, .offset= -128}}},
                                         .rank= 4,
                                         .dimensions=dimensions_layer4_0_conv2_weight,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer4_0_conv2_weight),
                                                        .dataSize=BINLEN(layer4_0_conv2_weight)}}}}}
  ), err);
  uint32_t dimensions_layer4_0_conv2_bias[] = {512};
  VALIDATE(resnet18_int8.addTensor("layer4_0_conv2_bias", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer4_0_conv2_bias",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0073393476195633f, .offset= -153}}},
                                         .rank= 1,
                                         .dimensions=dimensions_layer4_0_conv2_bias,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer4_0_conv2_bias),
                                                        .dataSize=BINLEN(layer4_0_conv2_bias)}}}}}
  ), err);

  /* ADDING NODE FOR layer4_0_conv2 */
  uint32_t dimensions_layer4_0_conv2_dilation[] = {2};
  uint32_t layer4_0_conv2_dilation[] = {1, 1};
  uint32_t dimensions_layer4_0_conv2_pad_amount[] = {2, 2};
  uint32_t layer4_0_conv2_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_layer4_0_conv2_stride[] = {2};
  uint32_t layer4_0_conv2_stride[] = {1, 1};
  Qnn_Param_t params_layer4_0_conv2[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer4_0_conv2_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer4_0_conv2_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer4_0_conv2_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer4_0_conv2_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_layer4_0_conv2_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer4_0_conv2_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer4_0_conv2_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer4_0_conv2_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer4_0_conv2_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_layer4_0_conv2[] = {
    "_286",
    "layer4_0_conv2_weight",
    "layer4_0_conv2_bias"
  };
  uint32_t dimensions__292[] = {32, 7, 7, 512};
  Qnn_Tensor_t outputs_layer4_0_conv2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_292",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0451336354017258f, .offset= -115}}},
            .rank= 4,
            .dimensions=dimensions__292,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "layer4_0_conv2", // Node Name
                                 "qti.aisw", // Package Name
                                 "Conv2d", // Qnn Node Type
                                 params_layer4_0_conv2, // Node Params
                                 4, // Num Node Params
                                 inputs_layer4_0_conv2, // Input Tensor Names
                                 3, // Num Input Tensor Names
                                 outputs_layer4_0_conv2, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);

  uint32_t dimensions_layer4_0_downsample_0_weight[] = {1, 1, 256, 512};
  VALIDATE(resnet18_int8.addTensor("layer4_0_downsample_0_weight", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer4_0_downsample_0_weight",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0058832294307649f, .offset= -128}}},
                                         .rank= 4,
                                         .dimensions=dimensions_layer4_0_downsample_0_weight,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer4_0_downsample_0_weight),
                                                        .dataSize=BINLEN(layer4_0_downsample_0_weight)}}}}}
  ), err);
  uint32_t dimensions_layer4_0_downsample_0_bias[] = {512};
  VALIDATE(resnet18_int8.addTensor("layer4_0_downsample_0_bias", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer4_0_downsample_0_bias",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0040180380456150f, .offset= -189}}},
                                         .rank= 1,
                                         .dimensions=dimensions_layer4_0_downsample_0_bias,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer4_0_downsample_0_bias),
                                                        .dataSize=BINLEN(layer4_0_downsample_0_bias)}}}}}
  ), err);

  /* ADDING NODE FOR layer4_0_downsample_0 */
  uint32_t dimensions_layer4_0_downsample_0_dilation[] = {2};
  uint32_t layer4_0_downsample_0_dilation[] = {1, 1};
  uint32_t dimensions_layer4_0_downsample_0_pad_amount[] = {2, 2};
  uint32_t layer4_0_downsample_0_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_layer4_0_downsample_0_stride[] = {2};
  uint32_t layer4_0_downsample_0_stride[] = {2, 2};
  Qnn_Param_t params_layer4_0_downsample_0[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer4_0_downsample_0_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer4_0_downsample_0_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer4_0_downsample_0_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer4_0_downsample_0_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_layer4_0_downsample_0_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer4_0_downsample_0_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer4_0_downsample_0_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer4_0_downsample_0_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer4_0_downsample_0_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_layer4_0_downsample_0[] = {
    "_277",
    "layer4_0_downsample_0_weight",
    "layer4_0_downsample_0_bias"
  };
  uint32_t dimensions__298[] = {32, 7, 7, 512};
  Qnn_Tensor_t outputs_layer4_0_downsample_0[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_298",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0278407987207174f, .offset= -123}}},
            .rank= 4,
            .dimensions=dimensions__298,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "layer4_0_downsample_0", // Node Name
                                 "qti.aisw", // Package Name
                                 "Conv2d", // Qnn Node Type
                                 params_layer4_0_downsample_0, // Node Params
                                 4, // Num Node Params
                                 inputs_layer4_0_downsample_0, // Input Tensor Names
                                 3, // Num Input Tensor Names
                                 outputs_layer4_0_downsample_0, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);


  /* ADDING NODE FOR module_add_6 */
  const char*  inputs_module_add_6[] = {
    "_292",
    "_298"
  };
  uint32_t dimensions__305[] = {32, 7, 7, 512};
  Qnn_Tensor_t outputs_module_add_6[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_305",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0335652530193329f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__305,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "module_add_6", // Node Name
                                 "qti.aisw", // Package Name
                                 "ElementWiseAdd", // Qnn Node Type
                                 nullptr, // Node Params
                                 0, // Num Node Params
                                 inputs_module_add_6, // Input Tensor Names
                                 2, // Num Input Tensor Names
                                 outputs_module_add_6, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);

  uint32_t dimensions_layer4_1_conv1_weight[] = {3, 3, 512, 512};
  VALIDATE(resnet18_int8.addTensor("layer4_1_conv1_weight", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer4_1_conv1_weight",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0020930483005941f, .offset= -128}}},
                                         .rank= 4,
                                         .dimensions=dimensions_layer4_1_conv1_weight,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer4_1_conv1_weight),
                                                        .dataSize=BINLEN(layer4_1_conv1_weight)}}}}}
  ), err);
  uint32_t dimensions_layer4_1_conv1_bias[] = {512};
  VALIDATE(resnet18_int8.addTensor("layer4_1_conv1_bias", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer4_1_conv1_bias",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0053378175944090f, .offset= -159}}},
                                         .rank= 1,
                                         .dimensions=dimensions_layer4_1_conv1_bias,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer4_1_conv1_bias),
                                                        .dataSize=BINLEN(layer4_1_conv1_bias)}}}}}
  ), err);

  /* ADDING NODE FOR layer4_1_conv1 */
  uint32_t dimensions_layer4_1_conv1_dilation[] = {2};
  uint32_t layer4_1_conv1_dilation[] = {1, 1};
  uint32_t dimensions_layer4_1_conv1_pad_amount[] = {2, 2};
  uint32_t layer4_1_conv1_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_layer4_1_conv1_stride[] = {2};
  uint32_t layer4_1_conv1_stride[] = {1, 1};
  Qnn_Param_t params_layer4_1_conv1[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer4_1_conv1_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer4_1_conv1_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer4_1_conv1_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer4_1_conv1_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_layer4_1_conv1_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer4_1_conv1_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer4_1_conv1_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer4_1_conv1_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer4_1_conv1_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_layer4_1_conv1[] = {
    "_305",
    "layer4_1_conv1_weight",
    "layer4_1_conv1_bias"
  };
  uint32_t dimensions__314[] = {32, 7, 7, 512};
  Qnn_Tensor_t outputs_layer4_1_conv1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_314",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0180877987295389f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__314,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "layer4_1_conv1", // Node Name
                                 "qti.aisw", // Package Name
                                 "Conv2d", // Qnn Node Type
                                 params_layer4_1_conv1, // Node Params
                                 4, // Num Node Params
                                 inputs_layer4_1_conv1, // Input Tensor Names
                                 3, // Num Input Tensor Names
                                 outputs_layer4_1_conv1, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);

  uint32_t dimensions_layer4_1_conv2_weight[] = {3, 3, 512, 512};
  VALIDATE(resnet18_int8.addTensor("layer4_1_conv2_weight", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer4_1_conv2_weight",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0021399301476777f, .offset= -128}}},
                                         .rank= 4,
                                         .dimensions=dimensions_layer4_1_conv2_weight,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer4_1_conv2_weight),
                                                        .dataSize=BINLEN(layer4_1_conv2_weight)}}}}}
  ), err);
  uint32_t dimensions_layer4_1_conv2_bias[] = {512};
  VALIDATE(resnet18_int8.addTensor("layer4_1_conv2_bias", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "layer4_1_conv2_bias",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0114674223586917f, .offset= -61}}},
                                         .rank= 1,
                                         .dimensions=dimensions_layer4_1_conv2_bias,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(layer4_1_conv2_bias),
                                                        .dataSize=BINLEN(layer4_1_conv2_bias)}}}}}
  ), err);

  /* ADDING NODE FOR layer4_1_conv2 */
  uint32_t dimensions_layer4_1_conv2_dilation[] = {2};
  uint32_t layer4_1_conv2_dilation[] = {1, 1};
  uint32_t dimensions_layer4_1_conv2_pad_amount[] = {2, 2};
  uint32_t layer4_1_conv2_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_layer4_1_conv2_stride[] = {2};
  uint32_t layer4_1_conv2_stride[] = {1, 1};
  Qnn_Param_t params_layer4_1_conv2[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer4_1_conv2_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer4_1_conv2_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer4_1_conv2_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer4_1_conv2_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_layer4_1_conv2_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer4_1_conv2_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "layer4_1_conv2_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_layer4_1_conv2_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)layer4_1_conv2_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_layer4_1_conv2[] = {
    "_314",
    "layer4_1_conv2_weight",
    "layer4_1_conv2_bias"
  };
  uint32_t dimensions__320[] = {32, 7, 7, 512};
  Qnn_Tensor_t outputs_layer4_1_conv2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_320",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.2487706243991852f, .offset= -60}}},
            .rank= 4,
            .dimensions=dimensions__320,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "layer4_1_conv2", // Node Name
                                 "qti.aisw", // Package Name
                                 "Conv2d", // Qnn Node Type
                                 params_layer4_1_conv2, // Node Params
                                 4, // Num Node Params
                                 inputs_layer4_1_conv2, // Input Tensor Names
                                 3, // Num Input Tensor Names
                                 outputs_layer4_1_conv2, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);


  /* ADDING NODE FOR module_add_7 */
  const char*  inputs_module_add_7[] = {
    "_320",
    "_305"
  };
  uint32_t dimensions__327[] = {32, 7, 7, 512};
  Qnn_Tensor_t outputs_module_add_7[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_327",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.2111094444990158f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__327,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "module_add_7", // Node Name
                                 "qti.aisw", // Package Name
                                 "ElementWiseAdd", // Qnn Node Type
                                 nullptr, // Node Params
                                 0, // Num Node Params
                                 inputs_module_add_7, // Input Tensor Names
                                 2, // Num Input Tensor Names
                                 outputs_module_add_7, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);


  /* ADDING NODE FOR avgpool */
  uint32_t dimensions_avgpool_filter_size[] = {2};
  uint32_t avgpool_filter_size[] = {7, 7};
  uint32_t dimensions_avgpool_pad_amount[] = {2, 2};
  uint32_t avgpool_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_avgpool_stride[] = {2};
  uint32_t avgpool_stride[] = {7, 7};
  Qnn_Param_t params_avgpool[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="filter_size",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "avgpool_filter_size",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_avgpool_filter_size,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)avgpool_filter_size,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "avgpool_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_avgpool_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)avgpool_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "avgpool_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_avgpool_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)avgpool_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="count_pad_for_edges",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_avgpool[] = {
    "_327"
  };
  uint32_t dimensions__330[] = {32, 1, 1, 512};
  Qnn_Tensor_t outputs_avgpool[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_330",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0705818310379982f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__330,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "avgpool", // Node Name
                                 "qti.aisw", // Package Name
                                 "PoolAvg2d", // Qnn Node Type
                                 params_avgpool, // Node Params
                                 4, // Num Node Params
                                 inputs_avgpool, // Input Tensor Names
                                 1, // Num Input Tensor Names
                                 outputs_avgpool, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);


  /* ADDING NODE FOR _330_nchw */
  uint32_t dimensions__330_nchw_perm[] = {4};
  uint32_t _330_nchw_perm[] = {0, 3, 1, 2};
  Qnn_Param_t params__330_nchw[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_330_nchw_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__330_nchw_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_330_nchw_perm,
                           .dataSize=16}}}}}}}
  };
  const char*  inputs__330_nchw[] = {
    "_330"
  };
  uint32_t dimensions__330_nchw[] = {32, 512, 1, 1};
  Qnn_Tensor_t outputs__330_nchw[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_330_nchw",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0705818310379982f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__330_nchw,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "_330_nchw", // Node Name
                                 "qti.aisw", // Package Name
                                 "Transpose", // Qnn Node Type
                                 params__330_nchw, // Node Params
                                 1, // Num Node Params
                                 inputs__330_nchw, // Input Tensor Names
                                 1, // Num Input Tensor Names
                                 outputs__330_nchw, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);

  uint32_t dimensions_fc_weight_permute[] = {1000, 512};
  VALIDATE(resnet18_int8.addTensor("fc_weight_permute", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "fc_weight_permute",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0056317872367799f, .offset= -128}}},
                                         .rank= 2,
                                         .dimensions=dimensions_fc_weight_permute,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(fc_weight_permute),
                                                        .dataSize=BINLEN(fc_weight_permute)}}}}}
  ), err);
  uint32_t dimensions_fc_bias[] = {1000};
  VALIDATE(resnet18_int8.addTensor("fc_bias", // Node Name
                                   (Qnn_Tensor_t) {
                                       .version= QNN_TENSOR_VERSION_1,
                                       {.v1= {
                                         .id=0,
                                         .name= "fc_bias",
                                         .type= QNN_TENSOR_TYPE_STATIC,
                                         .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                         .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                         .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                            {.scaleOffsetEncoding= {.scale= 0.0004361698229332f, .offset= -114}}},
                                         .rank= 1,
                                         .dimensions=dimensions_fc_bias,
                                         .memType= QNN_TENSORMEMTYPE_RAW,
                                         {.clientBuf= { .data=BINVARSTART(fc_bias),
                                                        .dataSize=BINLEN(fc_bias)}}}}}
  ), err);

  /* ADDING NODE FOR fc */
  const char*  inputs_fc[] = {
    "_330_nchw",
    "fc_weight_permute",
    "fc_bias"
  };
  uint32_t dimensions_class_probs[] = {32, 1000};
  Qnn_Tensor_t outputs_fc[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "class_probs",
            .type= QNN_TENSOR_TYPE_APP_READ,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.1869061440229416f, .offset= -63}}},
            .rank= 2,
            .dimensions=dimensions_class_probs,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(resnet18_int8.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                 "fc", // Node Name
                                 "qti.aisw", // Package Name
                                 "FullyConnected", // Qnn Node Type
                                 nullptr, // Node Params
                                 0, // Num Node Params
                                 inputs_fc, // Input Tensor Names
                                 3, // Num Input Tensor Names
                                 outputs_fc, // Output Tensors 
                                 1// Num Output Tensors 
  ), err);


  // Add all models to array to get graphsInfo
  QnnModel* models [] = {&resnet18_int8};
  uint32_t numModels = 1;

  // Populate the constructed graphs in provided output variables
  VALIDATE(getGraphInfoFromModels(*models, numModels, graphsInfo), err);
  *numGraphsInfo = numModels;

  return err;

} // PREPARE_GRAPHS

QNN_API
ModelError_t QnnModel_freeGraphsInfo(GraphInfoPtr_t** graphsInfo, uint32_t numGraphsInfo){
  return qnn_wrapper_api::freeGraphsInfo(graphsInfo, numGraphsInfo);
} // FREEGRAPHINFO

}