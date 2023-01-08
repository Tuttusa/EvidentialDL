
# EvidentialDL


Evidential Deep learning implementation Pytorch for regression and classification tasks based on :
- https://arxiv.org/pdf/1806.01768.pdf
- https://arxiv.org/pdf/1910.02600.pdf

## Installing

`pip install git+https://github.com/Tuttusa/EvidentialDL.git`

## Usage for classification
<pre>
from evidentialdl.layers import DenseDirichlet

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
</pre>

## Usage for regression
<pre>
from evidentialdl.layers import DenseNormalGamma

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
</pre>