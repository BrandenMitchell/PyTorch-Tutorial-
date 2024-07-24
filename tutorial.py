import torch
import numpy as np


#Tensors can be initialized directly from data 
data = [[1,3],[1,2]]
x_data = torch.tensor(data)

#can be created from a numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
#numpy arrays  have a set size on creation they also must have the same data type and therefore
#the same size in memory. They allow vectorizaton which lets you perform equations on all elements of a list without loops

#tensors can also be created from other tensors. the new tensor will retain the properties of the original (shape,Datatype)
#unless explicitly overridden

#shape is a tuple of tensor dimensions. It determines the dimentionality of the tensor
shape = (2,3,) #2 rows 3 cols
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zero_tensor = torch.zeros(shape)
print(f"random tensor: \n  {rand_tensor} \n ones Tensor:\n {ones_tensor} \n Zeros Tensor:\n {zero_tensor }")


#you can ask for the specific attributes of a tensor like size, datatype,device its running on  
#ex
tensor = torch.rand(3,4)
# print(f"Tensor size: \n {tensor.shape}")
# print(f"Tensor DataType: \n {tensor.dtype}")
# print(f"Tensor Device: \n {tensor.device}")

#tensor operation to move to the gpu if possible 
# if torch.cuda.is_available:
#     tensor = tensor.to('cuda')
#     print(f"device tensor is stored on: {tensor.device}")

