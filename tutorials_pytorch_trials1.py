import torch
import numpy as np

print("PyTorch version {}".format(torch.__version__))
print("GPU-enabled installation? {}".format(torch.cuda.is_available()))

t = torch.FloatTensor(2, 3)
print(t)
print(t.size())

t.zero_()

torch.FloatTensor([[1, 2, 3], [4, 5, 6]])

tl = torch.tensor([1, 2, 3])
t = torch.tensor([1., 2., 3.])
print("A 64-bit integer tensor: {}, {}".format(tl, tl.type()))
print("A 32-bit float tensor: {}, {}".format(t, t.type()))

t = torch.zeros(2, 3)
print(t)

t_zeros = torch.zeros_like(t)        # zeros_like returns a new tensor
t_ones = torch.ones(2, 3)            # creates a tensor with 1s
t_fives = torch.empty(2, 3).fill_(5) # creates a non-initialized tensor and fills it with 5
t_random = torch.rand(2, 3)          # creates a uniform random tensor
t_normal = torch.randn(2, 3)         # creates a normal random tensor

print(t_zeros)
print(t_ones)
print(t_fives)
print(t_random)
print(t_normal)

# creates a new copy of the tensor that is still linked to
# the computational graph (see below)
t1 = torch.clone(t)
assert id(t) != id(t1), 'Functional methods create a new copy of the tensor'

# Create a new multi-dimensional array in NumPy with the np datatype (np.float32)
a = np.array([1., 2., 3.])

# Convert the array to a torch tensor
t = torch.tensor(a)

print("NumPy array: {}, type: {}".format(a, a.dtype))
print("Torch tensor: {}, type: {}".format(t, t.dtype))

t = torch.randn(5, 6)
print(t)
i = torch.tensor([1, 3])
j = torch.tensor([4, 5])
print(t[i])                          # selects rows 1 and 3
print(t[i, j])

# Scalars =: creates a tensor with a scalar
# (zero-th order tensor,  i.e. just a number)
s = torch.tensor(42)
print(s)
s.item()

# Row vector
x = torch.randn(1,3)
print("Row vector\n{}\nwith size {}".format(x, x.size()))

# Column vector
v = torch.randn(3,1)
print("Column vector\n{}\nwith size {}".format(v, v.size()))

# Matrix
A = torch.randn(3, 3)
print("Matrix\n{}\nwith size {}".format(A, A.size()))

torch.sum(A)
A.sum()




