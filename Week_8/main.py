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
    for i in range(31):
        # 1000 labels per prediction
        pred = torch.nn.Softmax(preds[i*1000 : (i+1)*1000])
        if pred == labels[i]:
            correct += 1
    return correct/32


i = 0
i_avgs = 0
while os.path.exists(f"output/host_fp32/Result_{i}/class_probs.raw"):
    preds = np.fromfile(f"output/host_fp32/Result_{i}/class_probs.raw", dtype=np.float32)
    labels = pd.read_csv(f"/opt/data/imagenet/raw_test/batch_size_32/labels_{i}.csv").to_numpy()
    i += 1

    i_avgs += evaluate(preds, labels)

host_accuracy = i_avgs/i

j = 0
j_avgs = 0
while os.path.exists(f"output/cpu_fp32/Result_{j}/class_probs.raw"):
    preds = np.fromfile(f"output/cpu_fp32/Result_{j}/class_probs.raw", dtype=np.float32)
    labels = pd.read_csv(f"/opt/data/imagenet/raw_test/batch_size_32/labels_{j}.csv").to_numpy()
    j += 1

    j_avgs += evaluate(preds, labels)

cpu_accuracy = j_avgs/j

print(f" Accuracy on Host CPU: {host_accuracy}")
print(f" Accuracy on SDK CPU: {cpu_accuracy}")

# accuracy is zero in both cases, softmax yields much higher values than the labels.
# not sure how this happens.

