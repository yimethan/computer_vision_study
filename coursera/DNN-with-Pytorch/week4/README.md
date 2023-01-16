# Softmax function: using lines to classify data

## Softmax function

+ classification for multiple classes
+ uses cross-entropy loss
+ distributes probability throughout each output node
+ if binary classification, using sigmoid is same as softmax

```python
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader

def plot_data(data_set, model = None, n = 1, color = False):
    X = data_set[:][0]
    Y = data_set[:][1]
    plt.plot(X[Y == 0, 0].numpy(), Y[Y == 0].numpy(), 'bo', label = 'y = 0')
    plt.plot(X[Y == 1, 0].numpy(), 0 * Y[Y == 1].numpy(), 'ro', label = 'y = 1')
    plt.plot(X[Y == 2, 0].numpy(), 0 * Y[Y == 2].numpy(), 'go', label = 'y = 2')
    plt.ylim((-0.1, 3))
    plt.legend()
    if model != None:
        w = list(model.parameters())[0][0].detach()
        b = list(model.parameters())[1][0].detach()
        y_label = ['yhat=0', 'yhat=1', 'yhat=2']
        y_color = ['b', 'r', 'g']
        Y = []
        for w, b, y_l, y_c in zip(model.state_dict()['0.weight'], model.state_dict()['0.bias'], y_label, y_color):
            Y.append((w * X + b).numpy())
            plt.plot(X.numpy(), (w * X + b).numpy(), y_c, label = y_l)
        if color == True:
            x = X.numpy()
            x = x.reshape(-1)
            top = np.ones(x.shape)
            y0 = Y[0].reshape(-1)
            y1 = Y[1].reshape(-1)
            y2 = Y[2].reshape(-1)
            plt.fill_between(x, y0, where = y1 > y1, interpolate = True, color = 'blue')
            plt.fill_between(x, y0, where = y1 > y2, interpolate = True, color = 'blue')
            plt.fill_between(x, y1, where = y1 > y0, interpolate = True, color = 'red')
            plt.fill_between(x, y1, where = ((y1 > y2) * (y1 > y0)),interpolate = True, color = 'red')
            plt.fill_between(x, y2, where = (y2 > y0) * (y0 > 0),interpolate = True, color = 'green')
            plt.fill_between(x, y2, where = (y2 > y1), interpolate = True, color = 'green')
    plt.legend()
    plt.show()

class Data(Dataset):
    
    # Constructor
    def __init__(self):
        self.x = torch.arange(-2, 2, 0.1).view(-1, 1)
        self.y = torch.zeros(self.x.shape[0])
        self.y[(self.x > -1.0)[:, 0] * (self.x < 1.0)[:, 0]] = 1
        self.y[(self.x >= 1.0)[:, 0]] = 2
        self.y = self.y.type(torch.LongTensor)
        self.len = self.x.shape[0]
        
    # Getter
    def __getitem__(self,index):      
        return self.x[index], self.y[index]
    
    # Get Length
    def __len__(self):
        return self.len

data_set = Data()
data_set.x
plot_data(data_set)
```

### Build softmax classifier & Train model

```python
model = nn.Sequential(nn.Linear(1, 3))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
trainloader = DataLoader(dataset=data_set, batch_size=5)

LOSS = []

def train_model(epochs):

    for epoch in epochs:

        if epoch % 50 == 0:

            pass
            plot_data(data_set, model)

        for x, y in trainloader:

            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            LOSS.append(loss.item())
            loss.backward()
            optimizer.step()

train_model(300)
```

### Analyze results

```python
z = model(data_set.x)
_, yhat = z.max(1)

print('prediction:', yhat)

# acc of test set
correct = (data_set.y == yhat).sum().item()
acc = correct / len(data_set)

print('acc:', acc)
```

+ using softmax function to convert output to a probability

```python
# create softmax object
Softmax_fn=nn.Softmax(dim=-1)

# result is tensor Probability - each row corresponds to a different sample & each column corresponds to that sample belonging to a particular class
Probability = Softmax_fn

# first sample's prob
for i in range(3):

    print('prob of class {} is {}'.format(i, Probability[0, i]))
```

## MNIST using Softmax

+ MNIST images are 28 * 28

```python
!pip install torchvision==0.9.1 torch==1.8.1

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import numpy as np

def PlotParameters(model): 
    W = model.state_dict()['linear.weight'].data
    w_min = W.min().item()
    w_max = W.max().item()
    fig, axes = plt.subplots(2, 5)
    fig.subplots_adjust(hspace=0.01, wspace=0.1)
    for i, ax in enumerate(axes.flat):
        if i < 10:
            
            # Set the label for the sub-plot.
            ax.set_xlabel("class: {0}".format(i))

            # Plot the image.
            ax.imshow(W[i, :].view(28, 28), vmin=w_min, vmax=w_max, cmap='seismic')

            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
    plt.show()

def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(28, 28), cmap='gray')
    plt.title('y = ' + str(data_sample[1]))

train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
print("Print the training dataset:\n ", train_dataset)

validation_dataset = dsets.MNIST(root='./data', download=True, transform=transforms.ToTensor())
print("Print the validating dataset:\n ", validation_dataset)
```

### Build Softmax classifier

```python
class Softmax(nn.Module):

    def __init__(self, input_size, output_size):

        super(Softmax, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):

        z = self.linear(x)

        return x
```

### Define hyperparameters, cosf, optimizer & Train model

```python
input_dim = 28 * 28
output_dim = 10

model = SoftMax(input_dim, output_dim)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=100)
valloader = torch.utils.data.DataLoader(validation_dataset, batch_size=5000)

n_epochs = 10
LOSS = []
ACC = []

def train_model(epochs):

    for epoch in range(epochs):

        for x, y in trainloader:

            optimizer.zero_grad()

            yhat = model(x.view(-1, 28*28)) # flatten input

            loss = criterion(yhat, y)
            
            loss.backward()

            optimizer.step()

        correct = 0

        for x_test, y_test in valloader:

            pred = model(x_test.view(-1, 28*28))

            _, yhat = torch.max(z.data, 1)

            correct += (yhat == y_test).sum().item()

        accuracy = correct / len(validation_dataset)

        LOSS.append(loss.data)
        ACC .append(accuracy)

train_model(n_epochs)

# plot results
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(loss_list,color=color)
ax1.set_xlabel('epoch',color=color)
ax1.set_ylabel('total loss',color=color)
ax1.tick_params(axis='y', color=color)
    
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)  
ax2.plot( accuracy_list, color=color)
ax2.tick_params(axis='y', color=color)
fig.tight_layout()
```

## Hidden layers

+ in_size = input dimension
+ out_size = number of classes in the output = number of w and b respectively
+ no logistic function in the output
+ model.x.view(): converts rectangle tensors in a batch to a row tensor



7.5

+ back prop reduces number of computations
+ sigmoid function's problem: vanishing gradient