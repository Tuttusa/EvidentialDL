import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Sigmoid
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

import evidentialdl as edl
import torch.nn as nn
import torch.nn.functional as F
import torch

from evidentialdl.layers import DenseNormalGamma
from evidentialdl.losses.continuous import evidential_regression


def my_data(x_min, x_max, n, train=True):
    x = np.linspace(x_min, x_max, n)
    x = np.expand_dims(x, -1).astype(np.float32)

    sigma = 3 * np.ones_like(x) if train else np.zeros_like(x)
    y = x ** 3 + np.random.normal(0, sigma).astype(np.float32)

    return torch.from_numpy(x), torch.from_numpy(y)


def plot_predictions(x_train, y_train, x_test, y_test, y_pred, n_stds=4, kk=0):
    x_test = x_test[:, 0]
    mu, v, alpha, beta = y_pred.chunk(4, dim=-1)
    mu = mu[:, 0]
    var = np.sqrt(beta / (v * (alpha - 1)))
    var = np.minimum(var, 1e3)[:, 0]  # for visualization

    plt.figure(figsize=(5, 3), dpi=200)
    plt.scatter(x_train, y_train, s=1., c='#463c3c', zorder=0, label="Train")
    plt.plot(x_test, y_test, 'r--', zorder=2, label="True")
    plt.plot(x_test, mu, color='#007cab', zorder=3, label="Pred")
    plt.plot([-4, -4], [-150, 150], 'k--', alpha=0.4, zorder=0)
    plt.plot([+4, +4], [-150, 150], 'k--', alpha=0.4, zorder=0)

    for k in np.linspace(0, n_stds, 4):
        plt.fill_between(
            x_test, (mu - k * var), (mu + k * var),
            alpha=0.3,
            edgecolor=None,
            facecolor='#00aeef',
            linewidth=0,
            zorder=1,
            label="Unc." if k == 0 else None)

    plt.gca().set_ylim(-150, 150)
    plt.gca().set_xlim(-7, 7)
    plt.legend(loc="upper left")
    plt.show()

# %%
# Create some training and testing data
x_train, y_train = my_data(-4, 4, 1000)
x_test, y_test = my_data(-7, 7, 1000, train=False)

# %%
class MLP(nn.Module):
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(n_inputs, 64)
        self.layer2 = nn.Linear(64, 64)
        self.out = edl.layers.DenseNormalGamma(64, 1)

    # forward propagate input
    def forward(self, X):
        X = F.relu(self.layer1(X))
        X = F.relu(self.layer2(X))
        X = self.out(X)
        return X

model = MLP(x_train.shape[-1])

criterion = evidential_regression(coeff=1e-2)
optimizer = Adam(model.parameters(), lr=5e-4)

#%%
dataset = TensorDataset(x_train, y_train)
loader = DataLoader(dataset,batch_size=100)


#%%
# enumerate epochs
for epoch in range(700):
    # enumerate mini batches
    for i, (inputs, targets) in enumerate(loader):
        optimizer.zero_grad()
        # compute the model output
        yhat = model(inputs)
        # calculate loss
        loss = criterion(targets, yhat)
        # credit assignment
        loss.backward()
        # update model weights
        optimizer.step()
        print(loss.item())

# %%
with torch.no_grad():
    y_pred = model(x_test)
plot_predictions(x_train, y_train, x_test, y_test, y_pred)
