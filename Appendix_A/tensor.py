import torch
import numpy as np

# Check the version of PyTorch and if CUDA is available
print(torch.__version__)
print(torch.cuda.is_available())

# Understanding tensors

# create a 0D tensor (scalar) from a Python integer
tensor0d = torch.tensor(1)

# create a 1D tensor (vector) from a Python list
tensor1d = torch.tensor([1, 2, 3])

# create a 2D tensor (matrix) from a Python list
tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])

# create a 3D tensor from a Python list
tensor3d = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

# create a 3D tensor from a NumPy array
ary3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

# print the tensors
tensor3d_2 = torch.tensor(ary3d)
tensor3d_3 = torch.from_numpy(ary3d)

ary3d[0, 0, 0] = 999
print(tensor3d_2)
print(tensor3d_3)

