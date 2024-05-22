# conda activate /mnt/hd1/conda/aimet
import torch 
import aimet_common.defs
import aimet_torch.quantsim as qs
import MLP.mlp_model as mlp_model
import MLP.mlp_tester as mlp_tester
import MLP.dataloaders as dataloaders

# 10.2
# load MLP trained for section 4
model = mlp_model.MLP()
model.load_state_dict(torch.load("MLP/mpl.pth"))
model.eval()

print(" Check model loading")
print(model)
print("\n")

# test model loss for FP32
train, test = dataloaders.fashionMNIST(64)
loss, correct = mlp_tester.test(torch.nn.CrossEntropyLoss(), test, model)
print("FP32 model evaluation")
print(f"Accuracy: {correct}, CrossEntropyLoss: {loss}")
print("\n")

# Quantize model, standard quantization approch
# use first sample from the training dataset as dummy_input
# use standard post_training_tf as quantization_scheme
dummy_data, dummy_labels = train.dataset[0]
quant_mlp = qs.QuantizationSimModel(model = model,
                                      dummy_input = dummy_data,
                                      quant_scheme = aimet_common.defs.QuantScheme.post_training_tf,
                                      rounding_mode = "nearest",
                                      default_param_bw = 8,
                                      default_output_bw = 8)

# see documentation function for forward_pass_callback
def mlp_calibrate(io_model, i_use_cuda = False):
    batch_size = 64
    max_batch_number = 16 # 1024 samples

    train, test = dataloaders.fashionMNIST(batch_size=batch_size)
    
    # ensure model is in eval mode
    io_model.eval()
    current_batch_counter=0
    with torch.no_grad():
        for data, labels in train:
            current_batch_counter += 1
            io_model(data)

            if current_batch_counter == max_batch_number:
                break
# compute encodings, using mlp_calibrate for forward_pass_callback
quant_mlp.compute_encodings(forward_pass_callback = mlp_calibrate, forward_pass_callback_args = None)

# test quatized model
q_loss, q_correct = mlp_tester.test(torch.nn.CrossEntropyLoss(), test, quant_mlp.model)
print("\n")
print("Quantized model evaluation")
print(f"Accuracy: {q_correct}, CrossEntropyLoss: {q_loss}")
print("\n")

# compatre models
print("Difference between FP32 and quantized model")
print(f"Accuracy difference: {correct - q_correct}, Loss difference: {loss - q_loss}")
print("\n")

# check quantization
# thats just to much output:(
#print(quant_mlp)

try:
    quant_mlp.export(path = "/mnt/hd2/home/eml_02/EML_2024/Week_7/MLP/Quant_MLP", filename_prefix = "MLP_Net", dummy_input = dummy_data)
except:
    print("failed to save model")