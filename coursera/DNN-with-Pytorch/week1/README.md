# Torch tensors in 1D

```python
import torch
import numpy as np
import pandas as np
import matplotlib.pyplot as plt
% matplotlib inline

def plotVec(vectors):

    ax = plt.axes()

    for vec in vectors:

        ax.arrow(0, 0, *vec["vector"], head_width=0.05, color=vec["color"], head_length=0.1)
    
    plt.ylim(-2,2)
    plt.xlim(-2,2)
```

## Types and Shape

+ the elements in the list to be converted must have the same type

### torch.tensor()
  
+ int list converted to long tensor

```python
ints_to_tensor = torch.tensor([0, 1, 2, 3, 4])

print('The dtype of tensor object after converting it to tensor:', ints_to_tensor.dtype)
# torch.int64

print('The type of tensor object after converting it to tensor:', ints_to_tensor.type())
# torch.LongTensor

type(ints_to_tensor) # torch.Tensor
```

+ float list converted to float tensor

```python
floats_to_tensor = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])

print('The dtype of tensor object after converting it to tensor:', floats_to_tensor.dtype)
# torch.float32

print('The type of tensor object after converting it to tensor:', floats_to_tensor.type())
# torch.FloatTensor
```

### torch.FloatTensor()

+ convert int list to float tensor

```python
new_float_tensor = torch.FloatTensor([0, 1, 2, 3, 4])

print("The type of the new_float_tensor:", new_float_tensor.type())
# torch.FloatTensor
```

### tensor_obj.type(torch.FloatTensor)

+ convet existing tensor object to another tensor type

```python
old_int_tensor = torch.tensor([0, 1, 2, 3, 4])

new_float_tensor = old_int_tensor.type(torch.FloatTensor)

print("The type of the new_float_tensor:", new_float_tensor.type())
# torch.FloatTensor
```

### tensor_obj.size() and tensor_obj.ndimension()

```python
print("The size of the new_float_tensor: ", new_float_tensor.size())
# torch.Size([5])

print("The dimension of the new_float_tensor: ",new_float_tensor.ndimension())
# 1
```

### tensor_obj.view(row, col)

+ reshape tensor object [5] to [5, 1]
+ number of elements in a tensor must remain constant after applying view()

```python
twoD_float tensor = new_float_tensor.view(5, 1)

print("Original size:", new_float_tensor)
# tensor([0., 1., 2., 3., 4.])

print("Size after view method:", twoD_float_tensor)
# ([[0.],
#   [1.],
#   [2.],
#   [3.],
#   [4.]])
```

+ reshape tensor object with dynamic size

```python
twoD_float_tensor = new_float_tensor.view(-1, 1)

print("Original size:", new_float_tensor)
# tensor([0., 1., 2., 3., 4.])

print("Size after view method", twoD_float_tensor)
# ([[0.],
#   [1.],
#   [2.],
#   [3.],
#   [4.]])
```


### from_numpy()

```python
numpy_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
new_tensor = torch.from_numpy(numpy_array)

print("The dtype of new tensor:", new_tensor.dtype)
# torch.float64

print("The type of new tensor:", new_tensor.type())
# torch.DoubleTensor
```

### tensor_obj.numpy()

```python
back_to_numpy = new_tensor.numpy()

print("The numpy array from tensor:", back_to_numpy)
# [0. 1. 2. 3. 4.]

print("The dtype of numpy array:", back_to_numpy.dtype)
# float64
```

### item()

```python
this_tensor=torch.tensor([0,1, 2,3]) 

print("the first item is given by",this_tensor[0].item(),"the first tensor value is given by ",this_tensor[0])
# tensor(0)

print("the second item is given by",this_tensor[1].item(),"the second tensor value is given by ",this_tensor[1])
# tensor(1)

print("the third  item is given by",this_tensor[2].item(),"the third tensor value is given by ",this_tensor[2])
# tensor(2)
```

### tolist()

```python
torch_to_list=this_tensor.tolist()

print('tensor:', this_tensor,"\nlist:",torch_to_list)
# tensor([0, 1, 2, 3]) 
# list: [0, 1, 2, 3]
```

## Indexing and slicing

+ change value of tensor_sample's index 3~4
+ contain selected indexes
+ assign one value to selected indexes

```python
tensor_sample[3:5] = torch.tensor([300.0, 400.0])

print("Modified tensor:", tensor_sample)
# tensor([100,   1,   2, 300, 400])


selected_indexes = [3, 4]
subset_tensor_sample = tensor_sample[selected_indexes]

print("The subset of tensor_sample with the values on index 3 and 4: ", subset_tensor_sample)
# tensor([300, 400])


selected_indexes = [1, 3]
tensor_sample[selected_indexes] = 100000

print("Modified tensor with one value: ", tensor_sample)
# tensor([   100, 100000,      2, 100000,    400])
```

## Tensor functions

### Mean and Standard Deviation

```python
mean = math_tensor.mean()
print("The mean of math_tensor: ", mean)
# tensor(0.)

standard_deviation = math_tensor.std()
print("The standard deviation of math_tensor: ", standard_deviation)
# tensor(1.1547)
```

### Max and Min

```python
max_val = max_min_tensor.max()
print("Maximum number in the tensor: ", max_val)

min_val = max_min_tensor.min()
print("Minimum number in the tensor: ", min_val)
```

### Sin

```python
pi_tensor = torch.tensor([0, np.pi/2, np.pi])

sin = torch.sin(pi_tensor)

print("The sin result of pi_tensor: ", sin)
# tensor([ 0.0000e+00,  1.0000e+00, -8.7423e-08])
```

### Create Tensor by torch.linspace()

+ returns evenly spaced numbers over a specified interval

```python
len5_tensor = torch.linspace(-2, 2, steps=5)

print(len5_tensor)
# tensor([-2., -1.,  0.,  1.,  2.])
```

+ plot

```python
pi_tensor = torch.linspace(0, 2*np.pi, 100)
sin_result = torch.sin(pi_tensor)

plt.plot(pi_tensor.numpy(), sin_result.numpy())
```

## Tensor operations

### Addition and Subtraction

+ tensor + tensor

```python
u = torch.tensor([1, 0])
v = torch.tensor([0, 1])

w = u + v
print("The result tensor: ", w)
# tensor([1, 1])

plotVec([
    {"vector": u.numpy(), "name": 'u', "color": 'r'},
    {"vector": v.numpy(), "name": 'v', "color": 'b'},
    {"vector": w.numpy(), "name": 'w', "color": 'g'}
])
```

+ tensor + scalar

```python
u = torch.tensor([1, 2, 3, -1])
v = u + 1
print ("Addition Result: ", v)
```

### Multiplication

+ tensor * scalar

```python
u = torch.tensor([1, 2])

v = 2 * u

print("The result of 2 * u: ", v)
# tensor([2, 4])
```

+ tensor * tensor

```python
u = torch.tensor([1, 2])
v = torch.tensor([3, 2])

w = u * v

print ("The result of u * v", w)
# tensor([3, 4])
```

### Dot product - dot()

```python
u = torch.tensor([1, 2])
v = torch.tensor([3, 2])

print("Dot Product of u, v:", torch.dot(u,v))
```

# 2D Tensors

## Types and shape

```python
twoD_list = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
twoD_tensor = torch.tensor(twoD_list)
print("The New 2D Tensor: ", twoD_tensor)

print("The dimension of twoD_tensor: ", twoD_tensor.ndimension())
# 2

print("The shape of twoD_tensor: ", twoD_tensor.shape)
# torch.Size([3, 3])

print("The shape of twoD_tensor: ", twoD_tensor.size())
# torch.Size([3, 3])

print("The number of elements in twoD_tensor: ", twoD_tensor.numel())
# 9
```

## Indexing and Slicing

```python
tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])

print("1st-row first two columns? ", tensor_example[0, 0:2])
```

+ can't `tensor_obj[begin_row:end_row][begin_col:end_col]`
+ `tensor_obj[number: number][number]`

```python
tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])

sliced_tensor_example = tensor_example[1:3]

print("1. Slicing step on tensor_example: ")
print("Result after tensor_example[1:3]: ", sliced_tensor_example)
print("Dimension after tensor_example[1:3]: ", sliced_tensor_example.ndimension())
print("================================================")
print("2. Pick an index on sliced_tensor_example: ")
print("Result after sliced_tensor_example[1]: ", sliced_tensor_example[1])
print("Dimension after sliced_tensor_example[1]: ", sliced_tensor_example[1].ndimension())
print("================================================")
print("3. Combine these step together:")
print("Result: ", tensor_example[1:3][1])
print("Dimension: ", tensor_example[1:3][1].ndimension())
```

1. Slicing step on tensor_example: 
Result after tensor_example[1:3]:  tensor([[21, 22, 23],
        [31, 32, 33]])
Dimension after tensor_example[1:3]:  2
================================================
2. Pick an index on sliced_tensor_example: 
Result after sliced_tensor_example[1]:  tensor([31, 32, 33])
Dimension after sliced_tensor_example[1]:  1
================================================
3. Combine these step together:
Result:  tensor([31, 32, 33])
Dimension:  1

## Tensor operations

### Addition and subtraction

```python
X = torch.tensor([[1, 0],[0, 1]]) 
Y = torch.tensor([[2, 1],[1, 2]])

X_plus_Y = X + Y

print("The result of X + Y: ", X_plus_Y)
# tensor([[3, 1],
#        [1, 3]])
```

### Scalar multiplication

```python
Y = torch.tensor([[2, 1], [1, 2]])

two_Y = 2 * Y

print("The result of 2Y: ", two_Y)
# tensor([[4, 2],
#        [2, 4]])
```

### Element-wise Product/Hadamard Product

+ calculate `[[1, 0], [0, 1]]` * `[[2, 1], [1, 2]]`

```python
X = torch.tensor([[1, 0], [0, 1]])
Y = torch.tensor([[2, 1], [1, 2]])

X_times_Y = X * Y

print("The result of X * Y: ", X_times_Y)
# tensor([[2, 0],
#        [0, 2]])
```

### Matrix multiplication - torch.mm()

+ calculate multiplication between tensors with different sizes

```python
A = torch.tensor([[0, 1, 1], [1, 0, 1]])
B = torch.tensor([[1, 1], [1, 1], [-1, 1]])

A_times_B = torch.mm(A,B)

print("The result of A * B: ", A_times_B)
# tensor([[0, 2],
#        [0, 2]])
```

# Differenctiation in Pytorch

## Derivatives

+ create tensor `x` and set `requires_grad` to True
+ only Tensors of floating point and complex dtype can require gradients

```python
x = torch.tensor(2.0, requires_grad=True)

y = x ** 2
y.backward()

print("derivative at x=2:", x.grad)
```
### Custom autograd functions

+ by subclassing torch.autograd.Function and implementing forward & backward passes

```python
class SQ(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        # receive a Tensor containing input
        # and return a Tensor containing the output
        # ctx: context object that can be used to stash info for backward computation
        # can cache arbitrary objects for use in the backward pass using ctx.save_for_backward method

        result = i ** 2
        ctx.save_for_backward(i)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        # receive a Tensor containing gradient of the loss with respect to the output
        # need to compute gradient of the loss with respect to the input

        i, = ctx.saved_tensors
        grad_output = 2 * i

        return grad_output

# applying function

x = torch.tensor(2.0, requires_grad=True)
sq = SQ.apply

y = sq(x)
y.backward()
print(x.grad)
```

### Partial derivatives

```python
u = torch.tensor(1.0,requires_grad=True)
v = torch.tensor(2.0,requires_grad=True)

f = u * v + u ** 2

print("The result of v * u + u^2: ", f)
# tensor(3., grad_fn=<AddBackward0>)

f.backward()

print("The partial derivative with respect to u: ", u.grad)
# tensor(4.)
print("The partial derivative with respect to u: ", v.grad)
# tensor(1.)
```

+ derivative with respect to a function with multiple values
  + `torch.sum()`

```python
x = torch.linspace(-10, 10, 10, requires_grad = True)
Y = x ** 2
y = torch.sum(x ** 2)

y.backward()

plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label = 'derivative')
plt.xlabel('x')
plt.legend()
plt.show()
```

+ derivative of Relu activation function

```python
x = torch.linspace(-10, 10, 1000, requires_grad=True)
y = torch.relu(x)

y_sum = y.sum()
y.backward()
```

# Simple dataset

## Simple dataset

```python
# create dataset class

class toy_set(Dataset):
    
    # Constructor with defult values 
    def __init__(self, length = 100, transform = None):
        self.len = length
        self.x = 2 * torch.ones(length, 2)
        self.y = torch.ones(length, 1)
        self.transform = transform
     
    # Getter - indexing method
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)     
        return sample
    
    # Get Length - length method
    def __len__(self):
        return self.len



# create dataset object
our_dataset = toy_set()
print("Our toy_set object: ", our_dataset)
print("Value on index 0 of our toy_set object: ", our_dataset[0])
# (tensor([2., 2.]), tensor([1.]))
print("Our toy_set length: ", len(our_dataset))
# 100


# print out first 3 elements
for i in range(3):
    x, y=our_dataset[i]
    print("index: ", i, '; x:', x, '; y:', y)
# index:  0 ; x: tensor([2., 2.]) ; y: tensor([1.])
# index:  1 ; x: tensor([2., 2.]) ; y: tensor([1.])
# index:  2 ; x: tensor([2., 2.]) ; y: tensor([1.])

# dataset object is an iterable
# loop directly on the dataset object
for x,y in our_dataset:
    print(' x:', x, 'y:', y)
```

## Transforms

+ ex transform: x+1 and y*2

```python
# create tranform class

class add_mult(object):
    
    # Constructor
    def __init__(self, addx = 1, muly = 2):
        self.addx = addx
        self.muly = muly
    
    # Executor
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x + self.addx
        y = y * self.muly
        sample = x, y
        return sample



# create transform object
a_m = add_mult()
data_set = toy_set()


# assign outputs of original dataset to x and y
# apply transform to dataset and output values as x_ and y_
for i in range(10):

    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)

    x_, y_ = a_m(data_set[i])
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
```

+ apply transform whenever creating new toy_set object

```python
cust_dataset = toy_set(transform=a_m)

for i in range(10):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = cust_data_set[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
# original and transformed outputs are identical
```

## Compose

+ multiple transfroms on the dataset objects
  
```python
from torchvision import transforms

# create trasform class
class mult(object):
    
    # Constructor
    def __init__(self, mult = 100):
        self.mult = mult
        
    # Executor
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x * self.mult
        y = y * self.mult
        sample = x, y
        return sample


# combine transforms
data_transform = transforms.Compose([add_mult(), mult()])
print("The combination of transforms (Compose): ", data_transform)


compose_dataset = toy_set(transform=data_transform)
```

# Dataset and transforms

```python
# create dataset class
class Dataset(Dataset):

    # Constructor
    def __init__(self, csv_file, data_dir, transform=None):
        
        # Image directory
        self.data_dir=data_dir
        # The transform is goint to be used on image
        self.transform = transform

        data_dircsv_file=os.path.join(self.data_dir,csv_file)
        # Load the CSV file contians image info
        self.data_name= pd.read_csv(data_dircsv_file)
        
        # Number of images in dataset
        self.len=self.data_name.shape[0] 
    
    # Get the length
    def __len__(self):
        return self.len
    
    # Getter
    def __getitem__(self, idx):
        
        # Image file path
        img_name=os.path.join(self.data_dir,self.data_name.iloc[idx, 1])

        # Open image file
        image = Image.open(img_name)
        
        # The class label for the image
        y = self.data_name.iloc[idx, 0]
        
        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y



# torchvision transforms
# random vertical flip + to tensor
fliptensor_data_transform = transforms.Compose([transforms.RandomVerticalFlip(p=1), transforms.ToTensor()])



# create dataset object
dataset = Dataset(csv_file=csv_file , data_dir=directory,transform=fliptensor_data_transform)
```