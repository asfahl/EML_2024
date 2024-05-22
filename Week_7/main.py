# conda activate /mnt/hd1/conda/aimet
import torch 
import aimet_common.defs
import aimet_torch.quantsim as qs
import MLP.mlp_model as mlp_model
import MLP.mlp_tester as mlp_tester
import MLP.dataloaders as dataloaders

# 10.1.1
# compute_encodings
# Berechnet die optimale Quantisierung der Gewicht und Aktivierungsfkt. eines trainierten
# Models. Ein representativer Datensatz ist zusätzlich erforderlich.
#
# forward_pass_callback
# Der Parameter von compute_encodings ist eine Callback-Fkt. die die Forward-Fkt. des Modells auf einem gegebenen Datansatz aufruft.
# Auf Basis des durchlaufs der Forward Funktion errechnet compute_emcodings dann die Quantisierung.

# 10.2
# load MLP from Week 4
model = mlp_model.MLP()
model.load_state_dict(torch.load("MLP/mpl.pth"))
model.eval()

print(" Check model loading")
print(model)

# test model loss for FP32
train, test = dataloaders.fashionMNIST(64)
loss, correct = mlp_tester.test(torch.nn.CrossEntropyLoss(), test, model)
print("FP32 model evaluation")
print(f"Accuracy: {correct}, CrossEntropyLoss: {loss}")

# Quantize model
quant_mlp = qs.QuantizationSimModel(model = model,
                                      dummy_input = train,
                                      quant_scheme = aimet_common.defs.QuantScheme.post_training_tf,
                                      rounding_mode = "nearest",
                                      default_param_bw = 8,
                                      default_output_bw = 8)




