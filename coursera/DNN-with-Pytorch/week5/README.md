# Hidden Layer Deep Network: Sigmoid, Tanh, Relu activation functions

## NN Module & Training function

### Activation functions

+ sigmoid

```python
class Net(nn.Module):
    
    # Constructor
    def __init__(self, D_in, H1, H2, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    
    # Prediction
    def forward(self,x):
        x = torch.sigmoid(self.linear1(x)) 
        x = torch.sigmoid(self.linear2(x))
        x = self.linear3(x)
        return x
```

+ tanh

```python
class NetTanh(nn.Module):
    
    # Constructor
    def __init__(self, D_in, H1, H2, D_out):
        super(NetTanh, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    
    # Prediction
    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = self.linear3(x)
        return x
```

+ relu

```python
class NetRelu(nn.Module):

    # Constructor
    def __init__(self, D_in, H1, H2, D_out):
        
        super(NetRelu, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)

    # Prediction
    def forward(self, x):

        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)

        return x
```

### Train function

```python
def train(model, criterion, train_loader, val_loader, optimizer, epochs=100):

    i = 0
    useful_stuff = {'training_loss': [], 'val_acc': []}

    for epoch in range(epochs):

        for i, (x, y) in enumerate(train_loader):

            optimizer.zero_grad()
            z = model(x.view(-1, 28 * 28))

            loss = criterion(z, y)
            loss.backward()

            optimizer.step()

            useful_stuff['training_loss'].append(loss.data.item())

        correct = 0

        for x, y in val_loader:

            z = model(x.view(-1, 28 * 28))
            _, label = torch.max(z, 1)

            correct += (label == y).sum().item()

            acc = 100 * (correct / len(val_dataset))

            useful_stuff['val_acc'].append(acc)

        return useful_stuff
```

### Get dataset

```python
# MNIST
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
val_dataset = dsets.MNIST(root='./data'm, train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)
```

### Define network, Cost function, Optimizer & Train model

```python
criterion = nn.CrossEntropyLoss()

input_dim = 28 * 28
hidden_dim1 = 50
hidden_dim2 = 50
output_dim = 10

learning_rate = 0.01

# sigmoid
model = Net(input_dim, hidden_dim1, hidden_dim2, output_dim)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
training_results = train(model, criterion, train_loader, val_loader, optimizer, epochs=10)

# tanh
model_Tanh = NetTanh(input_dim, hidden_dim1, hidden_dim2, output_dim)
optimizer = torch.optim.SGD(model_Tanh.parameters(), lr=learning_rate)
training_results_tanh = train(model_Tanh, criterion, train_loader, val_loader, optimizer, epochs=10)

# relu
modelRelu = NetRelu(input_dim, hidden_dim1, hidden_dim2, output_dim)
optimizer = torch.optim.SGD(modelRelu.parameters(), lr=learning_rate)
training_results = train(modelRelu, criterion, train_loader, val_loader, optimizer, epochs=10)
```

### Analyzer results

```python
# compare training loss
plt.plot(training_results_tanh['training_loss'], label='tanh')
plt.plot(training_results['training_loss'], label='sigmoid')
plt.plot(training_results_relu['training_loss'], label='relu')
plt.ylabel('loss')
plt.title('training loss iterations')
plt.legend()

# compare validation loss
plt.plot(training_results_tanh['val_acc'], label='tanh')
plt.plot(training_results['val_acc'], label='sigmoid')
plt.plot(training_results_relu['training_loss'], label='relu')
plt.ylabel('loss')
plt.title('training loss iterations')
plt.legend()
```

# Deeper NN with nn.ModuleList()

```python
import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset, DataLoader
```

## NN Module and function for training

```python
class Net(nn.Module):

    def __init__(self, Layers):

        super(Net, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):

            self.hidden.append(nn.Linear(input_size, output_size))

    def forward(self, activation):

        L = len(self.hidden)

        for(l, linear_transform) in zip(range(L), self.hidden):

            if l < L - 1:

                activation = F.relu(linear_transform(activation))

            else:

                    activation = linear_transform(activation)

        return activation
```

```python
def accuracy(model, data_set):
    _, yhat = torch.max(model(data_set.x), 1)
    return (yhat == data_set.y).numpy().mean()

def train(dataset, model, criterion, train_loader, optimizer, epochs=100):

    LOSS, ACC = [], []

    for epoch in range(epochs):

        for x, y in train_loader:

            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            LOSS.append(loss.item())

        ACC.append(accuracy(model, dataset))

    return LOSS
```

## Train & Validation

```python
Layers = [2, 50, 3]

model = Net(Layers)

learning_rate = 0.10

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train_loader = DataLoader(dataset=data_set, batch_size=20)

criterion = nn.CrossEntropyLoss()

LOSS = train(data_set, model, criterion, train_loader, optimizer, epochs=100)
```

# Dropout for Classification

## Model, Optimizer, Cost function

```python
class Net(nn.Module):
    
    def __init__(self, in_size, n_hidden, out_size, p=0):

        super(Net, self).__init__()
        self.drop = nn.Dropout(p=p)
        self.linear1 = nn.Linear(in_size, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.linear3 = nn.Linear(n_hidden, out_size)
    
    def forward(self, x):

        x = F.relu(self.drop(self.linear1(x)))
        x = F.relu(self.drop(self.linear2(x)))
        x = self.linear3(x)
        return x

model = Net(2, 300, 2)
model_drop = Net(2, 300, 2, p=0.5)

optimizer_ofit = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer_drop = torch.optim.Adam(model_drop.parameters(), lr=0.01)

criterion = torch.nn.CrossEntropyLoss()
```

## Train

```python
# set model to train mode
model_drop.train()

epochs = 500

def train_model(epochs):
    
    for epoch in range(epochs):
        
        yhat = model(data_set.x)
        yhat_drop = model_drop(data_set.x)
        loss = criterion(yhat, data_set.y)
        loss_drop = criterion(yhat_drop, data_set.y)

        LOSS['training data no dropout'].append(loss.item())
        LOSS['validation data no dropout'].append(criterion(model(validation_set.x), validation_set.y).item())
        LOSS['training data dropout'].append(loss_drop.item())
        model_drop.eval()
        LOSS['validation data dropout'].append(criterion(model_drop(validation_set.x), validation_set.y).item())
        model_drop.train()

        optimizer_ofit.zero_grad()
        optimizer_drop.zero_grad()

        loss.backward()
        loss_drop.backward()

        optimizer_ofit.step()
        optimizer_drop.step()
        
train_model(epochs)
```

## Evaluate

```python
# set model to evaluation mode
model_drop.eval()

print("The accuracy of the model without dropout: ", accuracy(model, validation_set))
print("The accuracy of the model with dropout: ", accuracy(model_drop, validation_set))
```

# Dropout in Regression

## Model, Optimizer, Cost function

```python
class Net(nn.Module):
    
    # Constructor
    def __init__(self, in_size, n_hidden, out_size, p=0):
        super(Net, self).__init__()
        self.drop = nn.Dropout(p=p)
        self.linear1 = nn.Linear(in_size, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.linear3 = nn.Linear(n_hidden, out_size)
        
    def forward(self, x):
        x = F.relu(self.drop(self.linear1(x)))
        x = F.relu(self.drop(self.linear2(x)))
        x = self.linear3(x)
        return x

model = Net(1, 300, 1)
model_drop = Net(1, 300, 1, p=0.5)

optimizer_ofit = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer_drop = torch.optim.Adam(model_drop.parameters(), lr=0.01)

criterion = torch.nn.MSELoss()
```

## Train

```python
# model to train model
model_drop.train()

LOSS={['training data no dropout']=[],
    ['validation data no dropout']=[],
    ['training data dropout']=[],
    ['validation data dropout']=[]}

epochs = 500

def train_model(epochs):
    for epoch in range(epochs):

        yhat = model(data_set.x)
        yhat_drop = model_drop(data_set.x)
        loss = criterion(yhat, data_set.y)
        loss_drop = criterion(yhat_drop, data_set.y)

        LOSS['training data no dropout'].append(loss.item())
        LOSS['validation data no dropout'].append(criterion(model(validation_set.x), validation_set.y).item())
        LOSS['training data dropout'].append(loss_drop.item())
        model_drop.eval()
        LOSS['validation data dropout'].append(criterion(model_drop(validation_set.x), validation_set.y).item())
        model_drop.train()

        optimizer_ofit.zero_grad()
        optimizer_drop.zero_grad()

        loss.backward()
        loss_drop.backward()

        optimizer_ofit.step()
        optimizer_drop.step()
        
train_model(epochs)
```

## Evaluation

```python
model_drop.eval()

yhat = model(data_set.x)
yhat_drop = model_drop(data_set.x)

# plot
plt.figure(figsize=(6.1, 10))

plt.scatter(data_set.x.numpy(), data_set.y.numpy(), label="Samples")
plt.plot(data_set.x.numpy(), data_set.f.numpy(), label="True function", color='orange')
plt.plot(data_set.x.numpy(), yhat.detach().numpy(), label='no dropout', c='r')
plt.plot(data_set.x.numpy(), yhat_drop.detach().numpy(), label="dropout", c ='g')

plt.xlabel("x")
plt.ylabel("y")
plt.xlim((-1, 1))
plt.ylim((-2, 2.5))
plt.legend(loc = "best")
plt.show()
```

# Initialization

## Initializations

### Xavier

```python
class Net_Xavier(nn.Module):

    def __init__(self, Layers):

        super(Net_Xavier, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):

            linear = nn.Linear(input_size, output_size)
            torch.nn.init.xavier_uniform_(linear.weight)
            self.hidden.append(linear)

    def forward(self, x):

        L = len(self.hidden)

        for(l, linear_transform) in zip(range(L), self.hidden):

            if l < L - 1:

                x = torch.tanh(linear_transform(x))

            else:

                x = linear_transform(x)

        return x
```

### Uniform

```python
class Net_Uniform(nn.Module):

    def __init__(self, Layers):

        super(Net_Uniform, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):

            linear = nn.Linear(input_size, output_size)
            linear.weight.data.uniform_(0, 1)
            self.hidden.append(linear)

    def forward(self, x):

        L = len(self.hidden)

        for (l, linear_transform) in zip(range(L), self.hidden):

            if l < L - 1:

                x = torch.tanh(linear_transform(x))

            else:

                x = linear_transform(x)

        return x
```

### Pytorch default

```python
class Net(nn.Module):

    def __init__(self, Layers):

        super(Net, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):

            linear = nn.Linear(input_size, output_size)
            self.hidden.append(linear)

    def forward(self, x):

        L = len(self.hidden)

        for (l, linear_transform) in zip(rane(L), self.hidden):

            if l < L - 1:

                x = torch.tanh(linear_transform(x))

            else:

                x = linear_transform(x)

        return x
```

## Get data

```python
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)
```

## Network, cost, optimizer & Train model

```python
criterion = nn.CrossEntropyLoss()

input_dim = 28 * 28
output_dim = 10
layers = [input_dim, 100, 10, 100, 10, 100, output_dim]
epochs = 15

# default
model = Net(layers)
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
training_results = train(model, criterion, train_loader, validation_loader, optimizer, epochs=epochs)

# xavier
model_Xavier = Net_Xavier(layers)
optimizer = torch.optim.SGD(model_Xavier.parameters(), lr=learning_rate)
training_results_Xavier = train(model_Xavier, criterion, train_loader, validation_loader, optimizer, epochs=epochs)

# uniform
model_Uniform = Net_Uniform(layers)
optimizer = torch.optim.SGD(model_Uniform.parameters(), lr=learning_rate)
training_results_Uniform = train(model_Uniform, criterion, train_loader, validation_loader, optimizer, epochs=epochs)
```

## Results

```python
plt.plot(training_results_Xavier['training_loss'], label='Xavier')
plt.plot(training_results['training_loss'], label='Default')
plt.plot(training_results_Uniform['training_loss'], label='Uniform')
plt.ylabel('loss')
plt.xlabel('iteration ')  
plt.title('training loss iterations')
plt.legend()

plt.plot(training_results_Xavier['validation_accuracy'], label='Xavier')
plt.plot(training_results['validation_accuracy'], label='Default')
plt.plot(training_results_Uniform['validation_accuracy'], label='Uniform') 
plt.ylabel('validation accuracy')
plt.xlabel('epochs')   
plt.legend()
```

# Momentum

## Model

```python
class Net(nn.Module):

    def __init__(self, Layers):

        super(Net, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):

            L = len(self.hidden)

            for (l, linear_transform) in zip(range(L), self.hidden):

                if l < L - 1:

                    activation = F.relu(linear_transform(activation))

                else:

                    activation = linear_transform(activation)

            return activation
```

## Train

```python
def train(dataset, model, criterion, train_loader, optimizer, epochs=100):

    LOSS, ACC = [], []

    for epoch in range(epochs):

        for x, y in train_loader:

            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()

        LOSS.append(loss.item())
        ACC.append(accuracy(model, dataset))

    results = {'Loss': LOSS, 'Acc': ACC}

    fig, ax1 = plt.subplots()
    color ='tab:red'
    ax1.plot(LOSS,color=color)
    ax1.set_xlabel('epoch', color=color)
    ax1.set_ylabel('total loss', color=color)
    ax1.tick_params(axis = 'y', color=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(ACC, color=color)
    ax2.tick_params(axis='y', color=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    plt.show()

    return results


Results = {"momentum 0": {"Loss": 0, "Accuracy:": 0}, "momentum 0.1": {"Loss": 0, "Accuracy:": 0}}

# 1 hidden layer, 50 neurons, 0.5 momentum
Layers = [2, 50, 3]
model = Net(Layers)
learning_rate = 0.10
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
train_loader = DataLoader(dataset=data_set, batch_size=20)
criterion = nn.CrossEntropyLoss()

Results["momentum 0.5"] = train(data_set, model, criterion, train_loader, optimizer, epochs=100)
```

# Batch normalization

## with batch norm

```python
class NetBatchNorm(nn.Module):
    
    # Constructor
    def __init__(self, in_size, n_hidden1, n_hidden2, out_size):
        super(NetBatchNorm, self).__init__()
        self.linear1 = nn.Linear(in_size, n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, out_size)
        self.bn1 = nn.BatchNorm1d(n_hidden1)
        self.bn2 = nn.BatchNorm1d(n_hidden2)
        
    # Prediction
    def forward(self, x):
        x = self.bn1(torch.sigmoid(self.linear1(x)))
        x = self.bn2(torch.sigmoid(self.linear2(x)))
        x = self.linear3(x)
        return x
    
    # Activations, to analyze results 
    def activation(self, x):
        out = []
        z1 = self.bn1(self.linear1(x))
        out.append(z1.detach().numpy().reshape(-1))
        a1 = torch.sigmoid(z1)
        out.append(a1.detach().numpy().reshape(-1).reshape(-1))
        z2 = self.bn2(self.linear2(a1))
        out.append(z2.detach().numpy().reshape(-1))
        a2 = torch.sigmoid(z2)
        out.append(a2.detach().numpy().reshape(-1))
        return out
```

## without batch norm

```python
class Net(nn.Module):
    
    # Constructor
    def __init__(self, in_size, n_hidden1, n_hidden2, out_size):

        super(Net, self).__init__()
        self.linear1 = nn.Linear(in_size, n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, out_size)
    
    # Prediction
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = self.linear3(x)
        return x
    
    # Activations, to analyze results 
    def activation(self, x):
        out = []
        z1 = self.linear1(x)
        out.append(z1.detach().numpy().reshape(-1))
        a1 = torch.sigmoid(z1)
        out.append(a1.detach().numpy().reshape(-1).reshape(-1))
        z2 = self.linear2(a1)
        out.append(z2.detach().numpy().reshape(-1))
        a2 = torch.sigmoid(z2)
        out.append(a2.detach().numpy().reshape(-1))
        return out 
```

## Train

```python
def train(model, criterion, train_loader, val_loader, optimizer, epochs=100):

    i = 0
    useful_stuff = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):

        for i, (x, y) in enumerate(train_loader):

            model.train()
            optimizer.zero_grad()
            z = model(x.view(-1, 28 * 28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            useful_stuff['train_loss'].append(loss.data.item())

        correct = 0

        for x, y in val_loader:

            model.eval()
            yhat = model(x.view(-1, 28 * 28))
            _, label = torch.max(yhat, 1)
            correct += (label == y).sum().item()

        acc = 100 * (correct / len(val_dataset))
        useful_stuff['val_acc'].append(acc)

    return useful_stuff
```

## Train

### with batch norm

```python
model_norm  = NetBatchNorm(input_dim, hidden_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model_norm.parameters(), lr = 0.1)
training_results_Norm=train(model_norm , criterion, train_loader, validation_loader, optimizer, epochs=5)
```

### without batch norm

```python
model = Net(input_dim, hidden_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)
training_results = train(model, criterion, train_loader, validation_loader, optimizer, epochs=5)
```