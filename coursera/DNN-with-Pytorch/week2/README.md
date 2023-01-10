# LR 1D: Prediction

## Prediction

+ b = -1, w = 2
+ ŷ = -1 + 2x

```python
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-1.0, requires_grad=True)

# makes prediction
def forward(x):

    yhat = w * x + b

    return yhat

# prediction at x=1 and x=2
x = torch.tensor([[1.0], [2.0]])
yhat = forward(x)

print("prediction:", yhat)
# tensor([[1.],
#        [3.]], grad_fn=<AddBackward0>)
```

## Class Linear

### Linear()

```python
from torch.nn import Linear


# set random seed (params are randomly inited)
torch.manual_seed(1)

lr = Linear(in_features=1, out_feature=1, bias=True)

# params in torch.nn.Module accessed
print('w and b:', list(lr_parameters()))

# making prediction at x=[[1.0], [2.0]]
x = torch.tensor(([[1.0], [2.0]]))
yhat = lr(x)
```

### state_dict()

+ returns dictionary object corresponding to the layers of each param tensor

```python
print("Python dictionary: ",lr.state_dict())
# OrderedDict([('weight', tensor([[0.5153]])), ('bias', tensor([-0.4414]))])

print("keys: ",lr.state_dict().keys())
# odict_keys(['weight', 'bias'])

print("values: ",lr.state_dict().values())
# odict_values([tensor([[0.5153]]), tensor([-0.4414])])
```

```python
print("weight:",lr.weight)
# tensor([[0.5153]], requires_grad=True)

print("bias:",lr.bias)
# tensor([-0.4414], requires_grad=True)
```

## Build Custom Modules

```python
from torch import nn


# define linear regression class
class LR(nn.Module):

    # constructor
    def __init__(self, input_size, output_size):

        # inherit from parent
        super(LR, self).__init()
        self.linear = nn.Linear(input_size, output_size)

    # prediction function
    def forward(self, x):

        out = self.linear(x)

        return out


# create linear regression model
lr = LR(1, 1)


print('parameters:', list(lr.parameters()))
# [Parameter containing:
# tensor([[-0.1939]], requires_grad=True), Parameter containing:
# tensor([0.4694], requires_grad=True)]

print('linear model:', lr.linear)
# Linear(in_features=1, out_features=1, bias=True)
```

+ making prediction of samples

```python
x = torch.tensor([[1.0], [2.0]])
yhat = lr(x)
print(yhat)
```

+ params are also stored

```python
print("Python dictionary: ", lr.state_dict())
# OrderedDict([('linear.weight', tensor([[-0.1939]])), ('linear.bias', tensor([0.4694]))])

print("keys: ",lr.state_dict().keys())
# odict_keys(['linear.weight', 'linear.bias'])

print("values: ",lr.state_dict().values())
# odict_values([tensor([[-0.1939]]), tensor([0.4694])])
```

# LR 1D: Training One Parameter

```python
import numpy as np
import matplotlib.pyplot as plt

# class for visualizing data space and params
class plot_diagram():
    
    # Constructor
    def __init__(self, X, Y, w, stop, go = False):
        start = w.data
        self.error = []
        self.parameter = []
        self.X = X.numpy()
        self.Y = Y.numpy()
        self.parameter_values = torch.arange(start, stop)
        self.Loss_function = [criterion(forward(X), Y) for w.data in self.parameter_values] 
        w.data = start
        
    # Executor
    def __call__(self, Yhat, w, error, n):
        self.error.append(error)
        self.parameter.append(w.data)
        plt.subplot(212)
        plt.plot(self.X, Yhat.detach().numpy())
        plt.plot(self.X, self.Y,'ro')
        plt.xlabel("A")
        plt.ylim(-20, 20)
        plt.subplot(211)
        plt.title("Data Space (top) Estimated Line (bottom) Iteration " + str(n))
        plt.plot(self.parameter_values.numpy(), torch.tensor(self.Loss_function).numpy())   
        plt.plot(self.parameter, self.error, 'ro')
        plt.xlabel("B")
        plt.figure()
    
    # Destructor
    def __del__(self):
        plt.close('all')
```

## Make some data

```python
import torch

# generate data
X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = -3 * X
Y = f + 0.1 * torch.randn(X.size())

plt.plot(X.numpy(), Y.numpy(), 'rx', label = 'Y')

plt.plot(X.numpy(), f.numpy(), label = 'f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```

## Create model & Cost function

```python
def forward(x):

    return w * x # without bias
```

+ cost function using MSE(Mean Square Error)

```python
def criterion(yhat, y):

    return torch.mean((yhat-y) ** 2)
```

+ learning raet and LOSS to record loss of each iteration

```python
lr = 0.1
LOSS = []
```

+ model params

```python
w = torch.tensor(-10.0, requires_grad=True)
```

## Train the model

```python
def train_model(iter):
    for epoch in range (iter):
        
        # make prediction
        Yhat = forward(X)
        
        # cost of the iteration
        loss = criterion(Yhat,Y)
        
        # plot
        gradient_plot(Yhat, w, loss.item(), epoch)
        
        # store loss
        LOSS.append(loss.item())
        
        # compute gradient of the loss with respect to all the learnable parameters
        loss.backward()
        
        # updata parameters
        w.data = w.data - lr * w.grad.data
        
        # zero the gradients before running the backward pass
        w.grad.data.zero_()
```

+ running 4 iters of grad descent & plotting cost for each iter

```python
train_model(4)

plt.plot(LOSS)
plt.tight_layout()
plt.xlabel("Epoch/Iterations")
plt.ylabel("Cost")
```

# LR 1D: Training two parameters

```python
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits import mplot3d

# class for visualizing data/param space
class plot_error_surfaces(object):
    
    # Constructor
    def __init__(self, w_range, b_range, X, Y, n_samples = 30, go = True):
        W = np.linspace(-w_range, w_range, n_samples)
        B = np.linspace(-b_range, b_range, n_samples)
        w, b = np.meshgrid(W, B)    
        Z = np.zeros((30,30))
        count1 = 0
        self.y = Y.numpy()
        self.x = X.numpy()
        for w1, b1 in zip(w, b):
            count2 = 0
            for w2, b2 in zip(w1, b1):
                Z[count1, count2] = np.mean((self.y - w2 * self.x + b2) ** 2)
                count2 += 1
            count1 += 1
        self.Z = Z
        self.w = w
        self.b = b
        self.W = []
        self.B = []
        self.LOSS = []
        self.n = 0
        if go == True:
            plt.figure()
            plt.figure(figsize = (7.5, 5))
            plt.axes(projection='3d').plot_surface(self.w, self.b, self.Z, rstride = 1, cstride = 1,cmap = 'viridis', edgecolor = 'none')
            plt.title('Cost/Total Loss Surface')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.show()
            plt.figure()
            plt.title('Cost/Total Loss Surface Contour')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.contour(self.w, self.b, self.Z)
            plt.show()
    
    # Setter
    def set_para_loss(self, W, B, loss):
        self.n = self.n + 1
        self.W.append(W)
        self.B.append(B)
        self.LOSS.append(loss)
    
    # Plot diagram
    def final_plot(self): 
        ax = plt.axes(projection = '3d')
        ax.plot_wireframe(self.w, self.b, self.Z)
        ax.scatter(self.W,self.B, self.LOSS, c = 'r', marker = 'x', s = 200, alpha = 1)
        plt.figure()
        plt.contour(self.w,self.b, self.Z)
        plt.scatter(self.W, self.B, c = 'r', marker = 'x')
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()
    
    # Plot diagram
    def plot_ps(self):
        plt.subplot(121)
        plt.ylim
        plt.plot(self.x, self.y, 'ro', label="training points")
        plt.plot(self.x, self.W[-1] * self.x + self.B[-1], label = "estimated line")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim((-10, 15))
        plt.title('Data Space Iteration: ' + str(self.n))

        plt.subplot(122)
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c = 'r', marker = 'x')
        plt.title('Total Loss Surface Contour Iteration' + str(self.n))
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()
```

## Make some data

```python
import torch

x = torch.arange(-3, 3, 0.1).view(-1, 1)
f = 1 * x - 1
Y = f + 0.1 * torch.randn(X.size())

plt.plot(X.numpy(), Y.numpy(), 'rx', label = 'y')
plt.plot(X.numpy(), f.numpy(), label = 'f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
```

## Create model & Cost function (total loss)

```python
def forward(x):

    return w * x + b


def criterion(yhat, y):

    return torch.mean((yhat - y) ** 2)
```

## Train the model

```python
w = torch.tensor(-15.0, requires_grad=True)
b = torch.tensor(-10.0, requires_grad=True)

lr = 0.1
LOSS = []


# function for training
def train_model(iter):

    yhat = forward(X)
    loss = criterion(yhat, Y)

    get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
    if epoch % 3 == 0:
        get_surface.plot_ps()

    LOSS.append(loss.item())
    loss.backward()

    w.data = w.data - lr * w.grad.data
    b.data = b.data - lr * b.grad.data

    w.grad.data.zero_()
    b.grad.data.zero_()


# train model
train_model(15)

# plot loss result
get_surface.final_plot()
plt.plot(LOSS)
plt.tight_layout()
plt.xlabel("epoch/iter")
plt.ylabel("cost")
```

# LR 1D: Training 2 param SGD

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

# The class for plot the diagram

class plot_error_surfaces(object):
    
    # Constructor
    def __init__(self, w_range, b_range, X, Y, n_samples = 30, go = True):
        W = np.linspace(-w_range, w_range, n_samples)
        B = np.linspace(-b_range, b_range, n_samples)
        w, b = np.meshgrid(W, B)    
        Z = np.zeros((30, 30))
        count1 = 0
        self.y = Y.numpy()
        self.x = X.numpy()
        for w1, b1 in zip(w, b):
            count2 = 0
            for w2, b2 in zip(w1, b1):
                Z[count1, count2] = np.mean((self.y - w2 * self.x + b2) ** 2)
                count2 += 1
            count1 += 1
        self.Z = Z
        self.w = w
        self.b = b
        self.W = []
        self.B = []
        self.LOSS = []
        self.n = 0
        if go == True:
            plt.figure()
            plt.figure(figsize = (7.5, 5))
            plt.axes(projection = '3d').plot_surface(self.w, self.b, self.Z, rstride = 1, cstride = 1,cmap = 'viridis', edgecolor = 'none')
            plt.title('Loss Surface')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.show()
            plt.figure()
            plt.title('Loss Surface Contour')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.contour(self.w, self.b, self.Z)
            plt.show()
    
    # Setter
    def set_para_loss(self, W, B, loss):
        self.n = self.n + 1
        self.W.append(W)
        self.B.append(B)
        self.LOSS.append(loss)
    
    # Plot diagram
    def final_plot(self): 
        ax = plt.axes(projection = '3d')
        ax.plot_wireframe(self.w, self.b, self.Z)
        ax.scatter(self.W, self.B, self.LOSS, c = 'r', marker = 'x', s = 200, alpha = 1)
        plt.figure()
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c = 'r', marker = 'x')
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()
    
    # Plot diagram
    def plot_ps(self):
        plt.subplot(121)
        plt.ylim
        plt.plot(self.x, self.y, 'ro', label = "training points")
        plt.plot(self.x, self.W[-1] * self.x + self.B[-1], label = "estimated line")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim((-10, 15))
        plt.title('Data Space Iteration: ' + str(self.n))
        plt.subplot(122)
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c = 'r', marker = 'x')
        plt.title('Loss Surface Contour Iteration' + str(self.n))
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()
```

## Make some data

```python
# set random seed
torch.manual_seed(1)

X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = 1 * X - 1
Y = f + 0.1 * torc.randn(X.size())

plt.plot(X.numpy(), Y.numpy(), 'rx', label='y')
plot.plot(X.numpy(), f.nupy(), label='f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```

## Create model & Cost function

```python
def forward(x):

    return w * x + b

def criterion(yhat, y):

    return torch.mean((yhat - y) ** 2)
```

## Train the model

### Batch gradient descent

```python
get_surface = plot_error_surfaces(15, 13, x, y, 30)

w = torch.tensor(-15.0, requires_grad=True)
b = torch.tensor(-10.0, requires_grad=True)

lr = 0.1
LOSS_BGD = []

def train_model(iter):

    yhat = forward(x)
    loss = criterion(yhat, y).item()
    
    get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
    get_surface.plot_ps()

    LOSS_BGD.append(loss.item())
    loss.backward()

    w.data = w.data - w.grad.data
    b.data = b.data = b.grad.data

    w.grad.data.zero_()
    b.grad.data.zero_()

train_model(10)
```

### Stochastic gradient descent

```python
get_surface = plot_error_surfaces(15, 13, x, y, 30, go = False)

LOSS_SGD = []

w = torch.tensor(-15.0, requires_grad=True)
b = torch.tensor(-10.0, requires_grad=True)

def train_model_sgd(iter):

    for epoch in range(iter):

        # SGD is an approximation of out true total loss/cost, in this line of code we calculate our true loss/cost and store it
        Yhat =forward

        loss = criterion(Yhat, y)
        LOSS_SGD.append(loss.tolist())

        for x, y in zip(x, y):

            yhat = forward(x)
            loss = criterion(yhat, y).item()

            get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.item())

            loss.backward()

            w.data = w.data - w.grad.data
            b.data = b.data - b.grad.data

            w.grad.data.zero_()
            b.grad.data.zero_()

        get_surface.plot_pls()


train_model_SGD(10)
```

## SGD with Dataset DataLoader

```python
from torch.utils.data import Dataset, DataLoader

# create dataset
class Data(Dataset):

    # constructor
    def __init(self):

        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.y = 1 * self.x - 1
        self.len = self.x.shape[0]

    def __getitem__(self.index):

        return self.x[index], self.y[index]

    def __len__(self):

        return self.len


# create dataset object
dataset = Data()
print('the length of the dataset:', len(dataset))


get_surface = plot_error_surfaces(15, 13, x, y, 30, go = False)
```

### DataLoader

```python
w = torch.tensor(-15.0,requires_grad=True)
b = torch.tensor(-10.0,requires_grad=True)
LOSS_Loader = []


# create DataLoader object
trainloader = DataLoader(dataset=dataset, batch_size=1)


# define function for training
def train_model_DataLoader(epochs):

    for epoch in range(epochs):

        yhat = forward(x)

        Loss_Loader.append(criterion(yhat, y).item())

        for x, y in trainloader:

            yhat = forward(x)
            loss = criterion(yhat, y)
            
            get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())

            loss.backward()

            w.data = w.data - w.grad.data
            b.data = b.data - b.grad.data
            w.grad.data.zero_()
            b.grad.data.zero_()

        get_surface.plot_ps()

train_model_DataLoader(10)
```

# LR 1D: Training 2 parameter Mini-batch gradient descent

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# for visualizing diagrams
class plot_error_surfaces(object):
    
    # Constructor
    def __init__(self, w_range, b_range, X, Y, n_samples = 30, go = True):
        W = np.linspace(-w_range, w_range, n_samples)
        B = np.linspace(-b_range, b_range, n_samples)
        w, b = np.meshgrid(W, B)    
        Z = np.zeros((30, 30))
        count1 = 0
        self.y = Y.numpy()
        self.x = X.numpy()
        for w1, b1 in zip(w, b):
            count2 = 0
            for w2, b2 in zip(w1, b1):
                Z[count1, count2] = np.mean((self.y - w2 * self.x + b2) ** 2)
                count2 += 1
            count1 += 1
        self.Z = Z
        self.w = w
        self.b = b
        self.W = []
        self.B = []
        self.LOSS = []
        self.n = 0
        if go == True:
            plt.figure()
            plt.figure(figsize = (7.5, 5))
            plt.axes(projection = '3d').plot_surface(self.w, self.b, self.Z, rstride = 1, cstride = 1, cmap = 'viridis', edgecolor = 'none')
            plt.title('Loss Surface')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.show()
            plt.figure()
            plt.title('Loss Surface Contour')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.contour(self.w, self.b, self.Z)
            plt.show()
            
     # Setter
    def set_para_loss(self, W, B, loss):
        self.n = self.n + 1
        self.W.append(W)
        self.B.append(B)
        self.LOSS.append(loss)
    
    # Plot diagram
    def final_plot(self): 
        ax = plt.axes(projection = '3d')
        ax.plot_wireframe(self.w, self.b, self.Z)
        ax.scatter(self.W, self.B, self.LOSS, c = 'r', marker = 'x', s = 200, alpha = 1)
        plt.figure()
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c = 'r', marker = 'x')
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()
    
    # Plot diagram
    def plot_ps(self):
        plt.subplot(121)
        plt.ylim()
        plt.plot(self.x, self.y, 'ro', label = "training points")
        plt.plot(self.x, self.W[-1] * self.x + self.B[-1], label = "estimated line")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Data Space Iteration: '+ str(self.n))
        plt.subplot(122)
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c = 'r', marker = 'x')
        plt.title('Loss Surface Contour')
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()
```

## Make some data

```python
import torch

torch.manual_seed(1)

X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = 1 * X - 1
Y = f + 0.1 * torch.randn(X.size())

plt.plot(X.numpy(), Y.numpy(), 'rx', label = 'y')
plt.plot(X.numpy(), f.numpy(), label = 'f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```

## Create model & Cost function

```python
def forward(x):

    return w * x + b

def criterion(yhat, y):

    return torch.mean((yhat-y) ** 2)

get_surface = plot_error_surfaces(15, 13, X, Y, 30)
```

## Train (BGD)

```python
w = torch.tensor(-15.0, requires_grad=True)
b = torch.tensor(-10.0, requires_grad=True)
lr = 0.1
LOSS_BGD = []

def train_model_BGD(epochs):

    for epoch in range(epochs):

        yhat = forward(x)
        loss = criterion(yhat, y)
        LOSS_BGD.append(loss)
        get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
        get_surface.plot_ps()
        loss.backward()

        w.data = w.data - lr * w.grad.data
        b.data = b.data - lr * b.grad.data
        w.grad.data.zero_()
        b.grad.data.zero_()

train_model_BGD(10)
```

## SGD with Dataset DataLoader

```python
get_surface = plot_error_surfaces(15, 13, X, Y, 30, go = False)

from torch.utils.data import Dataset, DataLoader

# dataset class
class Data(dataset):

    # constructor
    def __init__(self):

        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.y = 1 * x - 1
        self.len = self.x.shape[0]

    def __getitem__(self, index):

        return self.x[index], self.y[index]

    def __len__(self):

        return self.len


# dataset & dataloader object
dataset = Data()
trainloader = DataLoader(dataset=dataset, batch_size=1)


# train
w = torch.tensor(-15.0, requires_grad = True)
b = torch.tensor(-10.0, requires_grad = True)
LOSS_SGD = []
lr = 0.1

def train_model_SGD(epochs):

    for epoch in rane(epochs):

        yhat = forward(x)

        get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), criterion(Yhat, Y).tolist())

        get_surface.plot_ps()

        LOSS_SGD.append(criterion(yhat, y).tolist())

        for x, y in trainloader:

            yhat = forward(x)

            loss = criterion(yhat, y)

            get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), criterion(Yhat, Y).tolist())
        
            loss.backward()

            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            w.grad.data.zero_()
            b.grad.data.zero_()
        get_surface.plot_ps()

train_model_SGD(10)
```

## Mini-batch grad descent (batch_size=5)

```python
get_surface = plot_error_surfaces(15, 13, X, Y, 30, go = False)

dataset = Data()
trainloader = DataLoader(dataset = dataset, batch_size = 5)

w = torch.tensor(-15.0, requires_grad = True)
b = torch.tensor(-10.0, requires_grad = True)
LOSS_MINI5 = []
lr = 0.1


def train_model_mini5(epochs):

    for epoch in range(epochs):

        yhat = forward(x)

        get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), criterion(Yhat, Y).tolist())
        get_surface.plot_ps()

        LOSS_MINI5.append(criterion(forward(x), y).tolist())

        for x, y in trainloader:

            yhat = forward(x)

            loss = criterion(yhat, y)

            get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), criterion(Yhat, Y).tolist())

            loss.backward()

            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            w.grad.data.zero_()
            b.grad.data.zero_()

train_model_mini5(10)
```

## Mini-batch grad descent (batch_size=10)

```python
get_surface = plot_error_surfaces(15, 13, X, Y, 30, go = False)

dataset = Data()
trainloader = DataLoader(dataset = dataset, batch_size = 10)

w = torch.tensor(-15.0, requires_grad = True)
b = torch.tensor(-10.0, requires_grad = True)
LOSS_MINI5 = []
lr = 0.1


def train_model_mini10(epochs):

    for epoch in range(epochs):

        yhat = forward(x)

        get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), criterion(Yhat, Y).tolist())
        get_surface.plot_ps()

        LOSS_MINI5.append(criterion(forward(x), y).tolist())

        for x, y in trainloader:

            yhat = forward(x)

            loss = criterion(yhat, y)

            get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), criterion(Yhat, Y).tolist())

            loss.backward()

            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            w.grad.data.zero_()
            b.grad.data.zero_()

train_model_mini10(10)
```

## Pytorch build-in functions

```python
from torch import nn, optim

class LR(nn.Module):

    # constructor
    def __init(self, input_size, output_size):

        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    # predict
    def forward(self, x):

        yhat = self.linear(x)

        return yhat
```

+ cost function

```python
criterion = nn.MSELoss()
```

+ optimizer object

```python
model = LR(1, 1)

optimizer = optim.SGD(model.parameters(), lr=0.01)

optimizer.state_dict()
# {'state': {},
#  'param_groups': [{'lr': 0.01,
#    'momentum': 0,
#    'dampening': 0,
#    'weight_decay': 0,
#    'nesterov': False,
#    'maximize': False,
#    'foreach': None,
#    'differentiable': False,
#    'params': [0, 1]}]}
```

+ customize w and b

```python
model.state_dict()['linear.weight'][0] = -15
model.state_dict()['linear.bias'][0] = -10
```

### training BGD

```python
def train_model_BGD(iter):

    for epoch in range(iter):

        for x,y in trainloader:

            yhat = model(x)

            loss = criterion(yhat, y)

            get_surface.set_para_loss(model, loss.tolist()) 

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        get_surface.plot_ps()


train_model_BGD(10)
```

# LR: Training & validation data

```python
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn,optim
from torch.utils.data import Dataset, DataLoader

class Data(Dataset):
    
    # Constructor
    def __init__(self, train = True):
            self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
            self.f = -3 * self.x + 1
            self.y = self.f + 0.1 * torch.randn(self.x.size())
            self.len = self.x.shape[0]
            
            #outliers 
            if train == True:
                self.y[0] = 0
                self.y[50:55] = 20
            else:
                pass
      
    # Getter
    def __getitem__(self, index):    
        return self.x[index], self.y[index]
    
    # Get Length
    def __len__(self):
        return self.len
```

## Create training/validation dataset

```python
train_data = Data()
val_data = Data(train = False)
```

## Create LR object, DataLodaer, Criterion function

```python
from torch import nn

class LR(nn.Module):

    def __init__(self, input_size, output_size):

        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x)

        yhat = self.linear(x)

        return yhat


# MSELoss function and dataloader
criterion = nn.MSELoss()

trainloader = DataLoader(train_data, batch_size=1)
```

## Different lr and data structures

### Storing results

```python
lr = [0.0001, 0.001, 0.01, 0.1]

train_error = torch.zeros(len(lr))
val_error = torch.zeros(len(lr))

MODELS = []
```

### Train different models w different hyperparams

```python
def train_model_with_lr(iter, lr_list):

    for i, lr in enumerate(lr_list):

        model= LR(1, 1)

        optimizer = optim.SGD(model.parameters(), lr=lr)

        for epoch in range(iter):

            for x, y in trainloader:

                yhat = model(x)
                loss = criterion(yhat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # train
        yhat = model(train_data.x)
        train_loss = criterion(yhat, train_data.y)
        train_error[i] = train_loss.item()

        # validation
        yhat = model(val_data.x)
        val_loss = criterion(yhat, val_Data.y)
        val_error[i] = val_loss.item()

        MODELS.append(model)

train_model_with_lr(10, lr)
```

## Plot results & predictions

### Results

```python
plt.semilogx(np.array(learning_rates), train_error.numpy(), label = 'training loss/total Loss')

plt.semilogx(np.array(learning_rates), validation_error.numpy(), label = 'validation cost/total Loss')

plt.ylabel('Cost\ Total Loss')
plt.xlabel('learning rate')
plt.legend()
plt.show()
```

### Predictions

```python
i = 0

for model, learning_rate in zip(MODELS, learning_rates):

    yhat = model(val_data.x)

    plt.plot(val_data.x.numpy(), yhat.detach().numpy(), label = 'lr:' + str(learning_rate))

    print('i', yhat.detach().numpy()[0:3])

plt.plot(val_data.x.numpy(), val_data.f.numpy(), 'or', label = 'validation data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```