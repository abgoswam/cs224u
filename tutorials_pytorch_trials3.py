import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import math

print("PyTorch version {}".format(torch.__version__))
print("GPU-enabled installation? {}".format(torch.cuda.is_available()))

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

M = 1200

# sample from the x axis M points
x = np.random.rand(M) * 2*math.pi

# add noise
eta = np.random.rand(M) * 0.01

# compute the function
y = np.sin(x) + eta

# plot
_ = plt.scatter(x,y)

x_train = torch.tensor(x[0:1000]).float().view(-1, 1).to(device)
y_train = torch.tensor(y[0:1000]).float().view(-1, 1).to(device)

x_test = torch.tensor(x[1000:]).float().view(-1, 1).to(device)
y_test = torch.tensor(y[1000:]).float().view(-1, 1).to(device)


class SineDataset(data.Dataset):
    def __init__(self, x, y):
        super(SineDataset, self).__init__()
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

sine_dataset = SineDataset(x_train, y_train)

sine_dataset_test = SineDataset(x_test, y_test)

sine_loader = torch.utils.data.DataLoader(
    sine_dataset, batch_size=32, shuffle=True)

sine_loader_test = torch.utils.data.DataLoader(
    sine_dataset_test, batch_size=32)


class SineModel(nn.Module):
    def __init__(self):
        super(SineModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 1))

    def forward(self, x):
        return self.network(x)


# declare the model
model = SineModel().to(device)

# define the criterion
criterion = nn.MSELoss()

# select the optimizer and pass to it the parameters of the model it will optimize
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 1000

# training loop
for epoch in range(epochs):
    for i, (x_i, y_i) in enumerate(sine_loader):
        y_hat_i = model(x_i)  # forward pass

        loss = criterion(y_hat_i, y_i)  # compute the loss and perform the backward pass

        optimizer.zero_grad()  # cleans the gradients
        loss.backward()  # computes the gradients
        optimizer.step()  # update the parameters

    if epoch % 20:
        plt.scatter(x_i.data.cpu().numpy(), y_hat_i.data.cpu().numpy())

