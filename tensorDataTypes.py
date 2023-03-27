import torch
import numpy as np

# Tensor Data Types

# x = torch.ones(2, 2)
# print(x)
# print(x.dtype)

# x = torch.ones(2, 2, dtype=torch.int8)
# print(x)
# print(x.dtype)

# Type Casting (Changing tensor data type)

# x = torch.ones(1, dtype=torch.uint8)
# print(x)
# print(x.dtype)

# x = x.type(torch.float32)
# print(x)
# print(x.dtype)

# Converting between numpy and torch tensors

# x = torch.rand(2, 2)
# print(x)
# print(x.dtype)

# y = x.numpy()
# print(y)
# print(y.dtype)

x = np.zeros((2, 2), dtype=np.float32)
print(x)
print(x.dtype)

y = torch.from_numpy(x)
print(y)
print(y.dtype)
