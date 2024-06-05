# abd devices
# adb -s [DEVICE SERIAL] shell
# adb ls 

# created repo in d4334008

# meassure accuracy
import torch
import numpy as np
import os.path
import pandas as pd


def evaluate(preds, labels):
    correct = 0
    for i in range(0,31):
        # 1000 labels per prediction
        pred = np.argmax(preds[(i)*1000 : (i+1)*1000])
        if pred == labels[i-1]:
            correct += 1
    return correct/32


i = 0
i_avgs = 0
while i < 10:
    preds = np.fromfile(f"output/host_fp32/Result_{i}/class_probs.raw", dtype=np.float32)
    labels = pd.read_csv(f"/opt/data/imagenet/raw_test/batch_size_32/labels_{i}.csv").to_numpy()
    i += 1

    i_avgs += evaluate(preds, labels)

host_accuracy = i_avgs/i

j = 0
j_avgs = 0
while j < 10:
    preds = np.fromfile(f"output/cpu_fp32/Result_{j}/class_probs.raw", dtype=np.float32)
    labels = pd.read_csv(f"/opt/data/imagenet/raw_test/batch_size_32/labels_{j}.csv").to_numpy()
    j += 1

    j_avgs += evaluate(preds, labels)

cpu_accuracy = j_avgs/j

k = 0
k_avgs = 0
while k < 10:
    preds = np.fromfile(f"output/gpu_fp32/Result_{k}/class_probs.raw", dtype=np.float32)
    labels = pd.read_csv(f"/opt/data/imagenet/raw_test/batch_size_32/labels_{k}.csv").to_numpy()
    k += 1

    k_avgs += evaluate(preds, labels)

gpu_accuracy = k_avgs/k

l = 0
l_avgs = 0
while l < 10:
    preds = np.fromfile(f"output/htp_int8/Result_{l}/class_probs.raw", dtype=np.float32)
    labels = pd.read_csv(f"/opt/data/imagenet/raw_test/batch_size_32/labels_{l}.csv").to_numpy()
    l += 1

    l_avgs += evaluate(preds, labels)

htp_accuracy = l_avgs/l

print(f" Accuracy on Host CPU: {host_accuracy}")
print(f" Accuracy on SDK CPU: {cpu_accuracy}")
print(f" Accuracy on SDK GPU: {gpu_accuracy}")
print(f" Accuracy on SDK HTP: {htp_accuracy}")

# Accuracy on Host CPU: 0.640625
# Accuracy on SDK CPU: 0.640625
# Accuracy on SDK GPU: 0.640625
# Accuracy on SDK HTP: 0.61875
# Quantization results in an accuracy drop