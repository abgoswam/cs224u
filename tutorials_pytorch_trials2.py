import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

print("PyTorch version {}".format(torch.__version__))
print("GPU-enabled installation? {}".format(torch.cuda.is_available()))


class MyCustomModule(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_output_classes):
        # call super to initialize the class above in the hierarchy
        super(MyCustomModule, self).__init__()

        # first affine transformation
        self.W = nn.Linear(n_inputs, n_hidden)

        # non-linearity (here it is also a layer!)
        self.f = nn.ReLU()

        # final affine transformation
        self.U = nn.Linear(n_hidden, n_output_classes)

    def forward(self, x):
        y = self.U(self.f(self.W(x)))
        return y

# set the network's architectural parameters
n_inputs = 3
n_hidden= 4
n_output_classes = 2

# instantiate the model
model = MyCustomModule(n_inputs, n_hidden, n_output_classes)

# create a simple input tensor
# size is [1,3]: a mini-batch of one example,
# this example having dimension 3
x = torch.FloatTensor([[0.3, 0.8, -0.4]])

# compute the model output by **applying** the input to the module
y = model(x)

# inspect the output
print(y)

# the true label (in this case, 2) from our dataset wrapped
# as a tensor of minibatch size of 1
y_gold = torch.tensor([1])

# our simple classification criterion for this simple example
criterion = nn.CrossEntropyLoss()

# forward pass of our model (remember, using apply instead of forward)
y = model(x)

# apply the criterion to get the loss corresponding to the pair (x, y)
# with respect to the real y (y_gold)
loss = criterion(y, y_gold)

# the loss contains a gradient function that we can use to compute
# the gradient dL/dw (gradient with respect to the parameters
# for a given fixed input)
print(loss)

