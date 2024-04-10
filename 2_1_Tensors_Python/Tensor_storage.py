import torch
import numpy as np
import ctypes

T0 = [[0, 1, 2], [3, 4, 5]]
T1 = [[6, 7, 8], [9, 10, 11]]
T2 = [[12, 13, 14], [15, 16, 17]]
T3 = [[18, 19, 20], [21, 22, 23]]
l = [T0, T1, T2, T3]
T = torch.tensor(l)

# tensor attributes
print(f"tensor size:{T.size()}")
print(f"tensor stride:{T.stride()}")
print(f"tensor layout:{T.layout}")
print(f"tensor device:{T.device}")
print(f"tensor dtype:{T.dtype}")

# tensor with dtype float32
l_tensor_float = T.clone().detach().type(torch.float32)
print(f"Dtype of the tensor cast to float32: {l_tensor_float.dtype}")

# tensor with second dimension fixed
l_tensor_fixed = l_tensor_float[:,0,:]
print(f"tensor size:{l_tensor_fixed.size()}")
print(f"tensor stride:{l_tensor_fixed.stride()}")
print(f"tensor layout:{l_tensor_fixed.layout}")
print(f"tensor device:{l_tensor_fixed.device}")
print(f"tensor dtype:{l_tensor_fixed.dtype}")

# Die 2. Dimension des Tensors ist fixiert, daher entfällt diese dimension aus Size und Stride.
# Die weiteren Attribute des Tensors verbleiben unverändert. 

#tensor with complex view
l_tensor_complex_view = l_tensor_float[::2,1,:]
print(f"tensor size:{l_tensor_complex_view.size()}")
print(f"tensor stride:{l_tensor_complex_view.stride()}")
print(f"tensor layout:{l_tensor_complex_view.layout}")
print(f"tensor device:{l_tensor_complex_view.device}")
print(f"tensor dtype:{l_tensor_complex_view.dtype}")
print(l_tensor_complex_view)

# Der View legt die ersten beiden Dimensionen des Tensors fest. In der letzten Dimension werden
# alle verbleibenden Einträge ausgeben (daher size 3). Der erste Eintrag (::2) wählt jeden 2. Eintrag
# in der ersten Dimension aus, der Stride wird demnach verdoppelt (6->1). Der zweite Eintrag fixiert die 
# zweite Dimension, sodass sie entfällt. Der dritte Eintrag iteriert über alle verbleibenden Zellen 

#contigous tensor
l_tensor_contig = l_tensor_complex_view.contiguous()
print(f"tensor size:{l_tensor_contig.size()}")
print(f"tensor stride:{l_tensor_contig.stride()}")
print(f"tensor layout:{l_tensor_contig.layout}")
print(f"tensor device:{l_tensor_contig.device}")
print(f"tensor dtype:{l_tensor_contig.dtype}")
print(l_tensor_contig)

# Der Stride ändert sich von (12,1) auf (3,1). Das liegt daran, dass l_tensor_complex_view weiterhin auf dem
# Speicher des Originaltensors bewegt und daher große Teile des Tensors übersprungen werden müssen.
# contigous() fügt die einzelnen auswahlen zu  einem neuen Tensor zusammen, in dem die ausgelassenen Einträge
# nicht mehr vorkommen und daher auch nicht übersprungen werden müssen.

# internal storage
# the pointer to the memory space
l_data_ptr = l_tensor_contig.data_ptr()
# the tensors stride
stride = l_tensor_contig.stride()
# the tensors dimensions ( to end iteration)
size = l_tensor_contig.size()

# target array
l_data_raw = np.zeros(size[0]*size[1])
# iterate the storage
for i in range(size[0]*size[1]):
    l_data_raw[i] = (ctypes.c_float).from_address(l_data_ptr+i*stride[0]+i*stride[1]).value

# raw array [3. 4. 5. 15. 16. 17.]
print(l_data_raw)
# flattend contig array tensor([ 3.,  4.,  5., 15., 16., 17.])
print(l_tensor_contig.flatten())

# beide Arrays/Tensoren werden als "geplättetes" Array dargestelt, wie sie im Speicher vorliegen