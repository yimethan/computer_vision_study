# Convolution

## Conv2d

```python
conv = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=3, stride=2, padding=1)

conv.state_dict()['weight'][0][0]=torch.tensor([[1.0,1.0],[1.0,1.0]])
conv.state_dict()['bias'][0]=0.0
conv.state_dict()
```

+ (output size) = (((input image size) - (kernel size)) / (stride)) + 1

## Activation function & Max pooling

```python
conv = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=3)

Z = conv(image)

# activation
A = torch.relu(Z)
# or
relu = nn.ReLU()
A = relu(Z)

# max pooling
max = nn.MaxPool2d(2, stride=1)
```

## Multiple channels

```python
# three output channels
conv1 = nn.Conv2d(in_channels=1, out_channels=3,kernel_size=3)
```

## Convolutional NN

### Model

```python
class CNN(nn.Module):

    def __init__(self, out_1=2, out_2=1):

        super(CNN, self).__init__()

        # first conv layer
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=2, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=1)

        # second conv layer
        self.cnn2 = n.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=2, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=1)

        # FC layer
        self.fc1 = nn.Linear(out_2 * 7 * 7, 2)

    def forward(self, x):

        # first conv layer
        x = self.cnn1(x)
        # activation
        x = torch.relu(x)
        # max pool
        x = self.maxpool1(x)

        # second conv layer
        x = self.cnn2(x)
        # activaton
        x = torch.relu(x)
        # max pool
        x = self.maxpool2(x)

        # flatten output
        x = x.view(x.size(0), -1)

        # fc
        x =self.fc1(x)

        return x

    def activations(self, x):

        z1=self.cnn1(x)
        a1=torch.relu(z1)
        out=self.maxpool1(a1)
        
        z2=self.cnn2(out)
        a2=torch.relu(z2)
        out=self.maxpool2(a2)
        out=out.view(out.size(0),-1)

        return z1, a1, z2, a2, out

model = CNN(2,1)
```

### Train

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=10)
val_loader=torch.utils.data.DataLoader(dataset=validation_dataset,batch_size=20)

LOSS, ACC = [], []

N_test = len(val_dataset)
cost = 0

for epoch in range(epochs):

    cost = 0

    for x, y in train_loader:

        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z, y)
        loss.backward()
        optimizer.step()
        cost += loss.item()
    
    LOSS.append(cost)

    correct = 0

    for x_test, y_test in val_loader:

        z = model(x_test)
        _, yhat = torch.max(z.data, 1)

        correct += (yhat == y_test).sum().item()

    acc = correct / N_test

    ACC.append(acc)
```

## CNN with Batch normalization

### Get data

```python
IMAGE_SIZE = 16

composed = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=composed)
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=composed)
```

### Network class

```python
class CNN(nn.Module):

    def __init__(self, out_1=16, out_2=32, n_classes=10):

        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(1, out_1, kernel_size=5, padding=2)
        self.conv1_bn = nn.BatchNorm2d(out_1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(out_1, out_2, kernel_size=5, stride=1, padding=2)
        self.conv2_bn = nn.BatchNorm2d(out_2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(out_2 * 4 * 4, n_classes)
        self.bn_fc1 = nn.BatchNorm1d(10)

    def forward(self, x):

        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x
```

### Train model

```python
def train_model(model, train_loader, val_loader, optimizer, epochs=4):

    N_test = len(validation_dataset)
    ACC, LOSS = [], []

    for epoch in range(epochs):

        for x, y in train_loader:

            model.train()
            optimizer.zero_Grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            LOSS.append(loss.data)

        correct = 0

        for x_test, y_test in val_loader:

            model.eval()
            z = model(x_test)
            _, yhat =torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()

        acc = correct / N_test
        ACC.append(acc)

    return ACC, LOSS

model = CNN_batch(out_1=16, out_2=32)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
val_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)

acc_list, loss_list=train_model(model=model,epochs=10,train_loader=train_loader,validation_loader=val_loader,optimizer=optimizer)
```