import torch
import torchvision
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot
import torchvision.transforms as T
from evidentialdl.layers import DenseDirichlet
import numpy as np
from evidentialdl.losses.discrete import dirichlet_loss

# %%
n_epochs = 4
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

mnist_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))])

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=True, download=True,
                               transform=mnist_transforms),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=False, download=True,
                               transform=mnist_transforms),
    batch_size=batch_size_test, shuffle=True)

# %%
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = DenseDirichlet(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)


network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

# %%
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    K= 10
    network.train()
    global_step = 0
    n_batches = len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        target = torch.from_numpy(to_categorical(target, num_classes=K))
        alpha, probs = network(data)
        loss = dirichlet_loss(target, alpha, K, global_step, K * n_batches)
        loss.backward()
        optimizer.step()
        global_step += 1
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './results/model.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            alpha, preds = network(data)
            test_loss += F.nll_loss(preds, target, size_average=False).item()
            pred = preds.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# %%
test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

# %%
mnist_data = torchvision.datasets.MNIST('.', download=True)
rotater = T.RandomRotation(degrees=(90, 90))
real_image = mnist_data[0][0]
rotated_image = rotater(mnist_data[0][0])

# pyplot.imshow(rotated_image, cmap="gray")
# pyplot.show()

with torch.no_grad():
    alpha1, preds1 = network(torch.unsqueeze(mnist_transforms(rotated_image), 0))
    alpha, preds = network(torch.unsqueeze(mnist_transforms(real_image), 0))

#%%
d_alpha1 = np.random.dirichlet(torch.squeeze(alpha1), 20)
d_alpha = np.random.dirichlet(torch.squeeze(alpha), 20)

#%%
pyplot.imshow(real_image, cmap="gray")
pyplot.show()