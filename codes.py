============================ simple nn - single data point =======================

!pip install torchmetrics


device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size=64

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torchmetrics import Precision, Recall, F1Score, Accuracy
from torchmetrics.classification import accuracy

# cross entropy 
def loss_fn(y_pred,y_true):
  e=0.0001
  v=-torch.sum(y_true*torch.log(y_pred+e))
  return v

#loading data FashionMNIST
train_data = datasets.FashionMNIST(
                root="data",
                train = True,
                transform = ToTensor(),
                download = True)

test_data = datasets.FashionMNIST(
                root="data",
                train=False,
                download=True,
                transform = ToTensor())

trainLoader = DataLoader(train_data,batch_size=64)
testLoader = DataLoader(test_data,batch_size=64)

for X, y in testLoader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

#Defining Neural Network
class mynet(nn.Module):
    def __init__(self):
        super(mynet,self).__init__()
        self.net_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784,10),
            nn.Softmax()
        )
    
    def forward(self,x):
        x = self.net_stack(x)
        return x

    

def get_model():
    model = mynet().to(device)
    return model

#training
def _train(trainloader,my_model,loss_fun,optimizer):
    size = len(trainloader)
    my_model.train()
    for batch , (X,y) in enumerate(trainloader):
        X, y = X.to(device), y.to(device)
        ypred = my_model(X)
        y= F.one_hot(y,10)
        #print(ypred.shape,y.shape)
        loss = loss_fun(ypred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def train(trainloader,my_model,epochs,learning_rate=1e-3):
    loss_fun = loss_fn
    optimizer = torch.optim.SGD(my_model.parameters(),lr=learning_rate,momentum=0.9)

    for i in range(epochs):
        print("running epoch ",i)
        _train(trainloader,my_model,loss_fun,optimizer)
    print("FINISHED TRAINING:")

#testing

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            tmp = torch.nn.functional.one_hot(y, num_classes= 10).to(device)
            pred = model(X)
            test_loss += loss_fn(pred, tmp).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    #precision_recall_fscore_support(y_ground, y_pred, average='macro')
    accuracy1 = Accuracy().to(device)
    print('Accuracy :', accuracy1(pred,y))
    precision = Precision(average = 'macro', num_classes = 10).to(device)
    print('precision :', precision(pred,y))

    recall = Recall(average = 'macro', num_classes = 10).to(device)
    print('recall :', recall(pred,y))
    f1_score = F1Score(average = 'macro', num_classes = 10).to(device)
    print('f1_score :', f1_score(pred,y))
    return accuracy1,precision, recall, f1_score

my_model = get_model()
print(my_model)
print(my_model.parameters())

#for single data point
x , y = train_data[0]
print(y)
y = torch.Tensor([y])
print(type(x),type(y))
print(y)
x , y = x.to(device) , y.to(device)
ypred = my_model(x)
print(loss_fn(ypred,torch.tensor(y)))

#training on train set 
train(trainLoader,my_model,10,1e-4)

#testing on test set
test(testLoader,my_model,loss_fn)

============================= simple nn ===============================
!pip install torchmetrics

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torchmetrics import Precision, Recall, F1Score, Accuracy
from torchmetrics.classification import accuracy

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size=64

#custom loss function


# cross entropy 
def loss_fn(y_pred,y_true):
  e=0.0001
  v=-torch.sum(y_true*torch.log(y_pred+e))
  return v

#loading Data FashionMNIST

train_data = datasets.FashionMNIST(
                root="data",
                train = True,
                transform = ToTensor(),
                download = True)

test_data = datasets.FashionMNIST(
                root="data",
                train=False,
                download=True,
                transform = ToTensor())

trainLoader = DataLoader(train_data,batch_size=64)
testLoader = DataLoader(test_data,batch_size=64)

for X, y in testLoader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Training  and Testing functions


def _train(trainloader,my_model,loss_fun,optimizer):
    size = len(trainloader)
    my_model.train()
    for batch , (X,y) in enumerate(trainloader):
        X, y = X.to(device), y.to(device)
        ypred = my_model(X)
        y= F.one_hot(y,10)
        #print(ypred.shape,y.shape)
        loss = loss_fun(ypred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def train(trainloader,my_model,epochs,learning_rate=1e-3):
    loss_fun = loss_fn
    optimizer = torch.optim.SGD(my_model.parameters(),lr=learning_rate,momentum=0.9)

    for i in range(epochs):
        print("running epoch ",i)
        _train(trainloader,my_model,loss_fun,optimizer)
    print("FINISHED TRAINING:")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            tmp = torch.nn.functional.one_hot(y, num_classes= 10).to(device)
            pred = model(X)
            test_loss += loss_fn(pred, tmp).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    #precision_recall_fscore_support(y_ground, y_pred, average='macro')
    accuracy1 = Accuracy().to(device)
    print('Accuracy :', accuracy1(pred,y))
    precision = Precision(average = 'macro', num_classes = 10).to(device)
    print('precision :', precision(pred,y))

    recall = Recall(average = 'macro', num_classes = 10).to(device)
    print('recall :', recall(pred,y))
    f1_score = F1Score(average = 'macro', num_classes = 10).to(device)
    print('f1_score :', f1_score(pred,y))
    return accuracy1,precision, recall, f1_score

# neural network


class mynet(nn.Module):
    def __init__(self,num_channels,num_classes):
        super(mynet,self).__init__()
        self.net_stack = nn.Sequential(
            nn.Conv2d(in_channels = num_channels, out_channels=20,kernel_size=(5,5)),
            nn.ReLU(),
            nn.Conv2d(in_channels = 20, out_channels=50,kernel_size=(5,5)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(20000,10),
            nn.Softmax()
        )
    
    def forward(self,x):
        x = self.net_stack(x)
        return x

my_model2 = mynet(1,10)


print(my_model2)
print(my_model2.parameters())



# training on trainset with custom loss

train(trainLoader,my_model2,2,1e-4)

# testing on test set

test(testLoader,my_model2,loss_fn)

==============================  complex nn ============================


!pip install torchmetrics


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torchmetrics import Precision, Recall, F1Score, Accuracy
from torchmetrics.classification import accuracy

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size=64

#custom loss function


# cross entropy 
def loss_fn(y_pred,y_true):
  e=0.0001
  v=-torch.sum(y_true*torch.log(y_pred+e))
  return v

#loading Data FashionMNIST

train_data = datasets.FashionMNIST(
                root="data",
                train = True,
                transform = ToTensor(),
                download = True)

test_data = datasets.FashionMNIST(
                root="data",
                train=False,
                download=True,
                transform = ToTensor())

trainLoader = DataLoader(train_data,batch_size=64)
testLoader = DataLoader(test_data,batch_size=64)

for X, y in testLoader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Training  functions



def _train(trainloader,my_model,loss_fun,optimizer):
    num_data_points = len(trainloader.dataset)
    num_batches = len(trainloader)
    num_classes = len(trainloader.dataset.classes)
    my_model.train()
    train_loss = 0
    for batch , (X,y) in enumerate(trainloader):
        X, y = X.to(device), y.to(device)
        ypred = my_model(X)
        y= F.one_hot(y,num_classes)
        loss = loss_fun(ypred,y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

  
    print(f"Avg loss: {train_loss/num_batches:>8f} \n")

def train(trainloader,my_model,epochs,learning_rate=1e-3):
    loss_fun = loss_fn
    optimizer = torch.optim.SGD(my_model.parameters(),lr=learning_rate,momentum=0.9)

    for i in range(epochs):
        print("running epoch ",i)
        _train(trainloader,my_model,loss_fun,optimizer)

#Testing Function



def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    num_classes= len(dataloader.dataset.classes)
    model.eval()
    test_loss, correct = 0, 0

    ypred = []
    ytrue = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            tmp = F.one_hot(y, num_classes= num_classes).to(device)
            pred = model(X)
            ypred.append(pred.argmax(1))
            ytrue.append(y)
            test_loss += loss_fn(pred, tmp).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    ypred = torch.cat(ypred)
    ytrue = torch.cat(ytrue)
    
    
    print(f"\nAvg loss: {test_loss:>8f} \n")
    
    accuracy1 = Accuracy().to(device)
    print(f'Accuracy : {accuracy1(ypred,ytrue).item()*100:.3f} %')
    
    precision = Precision(average = 'macro', num_classes = num_classes).to(device)
    print(f'precision : {precision(ypred,ytrue).item() :.4f}')

    recall = Recall(average = 'macro', num_classes = num_classes).to(device)
    print(f'recall : {recall(ypred,ytrue).item():.4f}')
    
    f1_score = F1Score(average = 'macro', num_classes = num_classes).to(device)
    print(f'f1_score : {f1_score(ypred,ytrue).item(): .4f}')

# neural network


class mynet(nn.Module):
    def __init__(self,num_channels,num_classes,h,w,config):
        super(mynet,self).__init__()
        
        self.net_stack = nn.Sequential()
        self.num_params_first_fc = 0
        for i in range(len(config)):
            layer = config[i]

            num_in_channels=layer[0]
            num_out_channels=layer[1]
            kernel = layer[2]
            padding = layer[4] # assuming that padding is always given as 'same'
            
            if isinstance(layer[3],int):
                stride = [layer[3],layer[3]]
            else:
                stride = layer[3]

            self.net_stack.append(nn.Conv2d(in_channels = num_in_channels, out_channels=num_out_channels,
                                    kernel_size=kernel,stride = stride,padding=padding).to(device))
            self.net_stack.append(nn.ReLU())

            if padding != "same":
                if isinstance(padding,int):
                    padding = [padding,padding]
                h = (int)((h - kernel[0])/stride[0])+1
                w = (int)((w - kernel[1])/stride[1])+1

            if i == len(config)-1:
                self.num_params_first_fc = num_out_channels

        self.net_stack.append(nn.Flatten())
        self.net_stack.append(nn.Linear(self.num_params_first_fc*h*w,num_classes).to(device))
        self.net_stack.append(nn.Softmax(1))
    
    def forward(self,x):
        x = self.net_stack(x)
        return x

# get trained model function


def get_model(trainloader,config,epochs,learning_rate):
    
    N , num_channels , height , width = next(iter(trainloader))[0].shape
    num_classes = len(trainloader.dataset.classes)
    
    my_model = mynet(num_channels,num_classes,height,width,config).to(device)
    
    train(trainloader,my_model,epochs,learning_rate)
    
    return my_model

# get model with config parameters

config = [(1,20,(5,5),1,'same'), (20,40,(5,5),1,'same'),(40,60,(5,5),1,'same')]
my_model = get_model(trainLoader,config,epochs=5,learning_rate=1e-4)
print(my_model)

# testing

test(testLoader,my_model,loss_fn)

============================ code in hub config =====================================

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as Fun
from torchmetrics import Precision, Recall, F1Score, Accuracy
from torchmetrics.classification import accuracy


device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size=64



class myCnn(nn.Module):
    def __init__(self,num_channels,num_classes,h,w,config):
        super(myCnn,self).__init__()
        
        self.net_stack = nn.Sequential()
        self.final_output_channels = 0
        for i in range(len(config)):
            layer = config[i]

            num_in_channels=layer[0]
            num_out_channels=layer[1]
            kernel = layer[2]
            padding = layer[4] # assuming that padding is always given as 'same'
            
            if isinstance(layer[3],int):
                stride = [layer[3],layer[3]]
            else:
                stride = layer[3]

            self.net_stack.append(nn.Conv2d(in_channels = num_in_channels, out_channels=num_out_channels,
                                    kernel_size=kernel,stride = stride,padding=padding).to(device))
            self.net_stack.append(nn.ReLU())

            if padding != "same":
                if isinstance(padding,int):
                    padding = [padding,padding]
                h = (int)((h - kernel[0] + 2 *padding[0])/stride[0])+1
                w = (int)((w - kernel[1] + 2 *padding[0])/stride[1])+1

            if i == len(config)-1:
                self.final_output_channels = num_out_channels

        self.net_stack.append(nn.Flatten())
        self.net_stack.append(nn.Linear(self.final_output_channels*h*w,num_classes).to(device))
        self.net_stack.append(nn.Softmax(1))
    
    def forward(self,x):
        x = self.net_stack(x)
        return x


def custom_loss(ypred,ytrue):
    ypred , ytrue = ypred.to(device) , ytrue.to(device)
    v = -(ytrue * torch.log(ypred + 0.0001))
    v = torch.sum(v)
    return v


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    num_classes= len(dataloader.dataset.classes)
    model.eval()
    test_loss, correct = 0, 0

    ypred = []
    ytrue = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            tmp = Fun.one_hot(y, num_classes= num_classes).to(device)
            pred = model(X)
            ypred.append(pred.argmax(1))
            ytrue.append(y)
            test_loss += loss_fn(pred, tmp).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    ypred = torch.cat(ypred)
    ytrue = torch.cat(ytrue)
    
    
    print(f"\nAvg loss: {test_loss:>8f} \n")
    
    accuracy1 = Accuracy().to(device)
    acc = accuracy1(ypred,ytrue).item()*100
    print(f'Accuracy : {acc:.3f} %')
    
    precision = Precision(average = 'macro', num_classes = num_classes).to(device)
    pre = precision(ypred,ytrue).item()
    print(f'precision : {pre :.4f}')

    recall = Recall(average = 'macro', num_classes = num_classes).to(device)
    re = recall(ypred,ytrue).item()
    print(f'recall : {re:.4f}')
    
    f1_score = F1Score(average = 'macro', num_classes = num_classes).to(device)
    f1 = f1_score(ypred,ytrue).item()
    print(f'f1_score : {f1: .4f}')

    return acc,pre,re,f1


def _train(trainloader,my_model,loss_fun,optimizer):
    num_data_points = len(trainloader.dataset)
    num_batches = len(trainloader)
    num_classes = len(trainloader.dataset.classes)
    my_model.train()
    train_loss = 0
    for batch , (X,y) in enumerate(trainloader):
        X, y = X.to(device), y.to(device)
        ypred = my_model(X)
        y= Fun.one_hot(y,num_classes)
        loss = loss_fun(ypred,y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % (num_batches // 4) == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{num_data_points:>5d}]")
    print(f"Avg loss: {train_loss/num_batches:>8f} \n")


def train(trainloader,my_model,epochs,learning_rate=1e-3):
    loss_fun = custom_loss
    optimizer = torch.optim.SGD(my_model.parameters(),lr=learning_rate,momentum=0.9)

    for i in range(epochs):
        print("running epoch ",i)
        _train(trainloader,my_model,loss_fun,optimizer)


def get_model(trainloader,config,epochs,learning_rate):
    
    N , num_channels , height , width = next(iter(trainloader))[0].shape
    num_classes = len(trainloader.dataset.classes)
    
    my_model = myCnn(num_channels,num_classes,height,width,config).to(device)
    
    train(trainloader,my_model,epochs,learning_rate)
    
    return my_model


def get_data_loaders():
	train_data = datasets.FashionMNIST(
                root="data",
                train = True,
                transform = ToTensor(),
                download = True)
	test_data = datasets.FashionMNIST(
                root="data",
                train=False,
                download=True,
                transform = ToTensor())
	trainLoader = DataLoader(train_data,batch_size=batch_size)
	testLoader = DataLoader(test_data,batch_size=batch_size)

	return trainLoader,testLoader

def get_loss_func():
	 loss = custom_loss

	 return loss
   
   
=================================== code in colab =============================


import torch

!pip install torchmetrics

myrepo = 'cs21m007iittp/islcourse:practice'

entrypoints = torch.hub.list(myrepo,force_reload=True)

print (entrypoints)

trainingloader, testloader = torch.hub.load(myrepo,'get_data_loaders',force_reload=True)

print(len(trainingloader))
print(len(trainingloader.dataset))
print(len(next(iter(trainingloader))[0]))

config = [(1,20,(5,5),1,1), (20,40,(5,5),1,0),(40,60,(5,5),1,0)]

my_model = torch.hub.load(myrepo,'get_model',trainingloader,config,10,1e-4)

loss_fun = torch.hub.load(myrepo,'get_loss_func')

print(loss_fun)

result = torch.hub.load(myrepo,'test',testloader,my_model,loss_fun)

print(result)

====================================== conv2d output =================================

import numpy as np

def conv2d_output_size(input_size, out_channels, padding, kernel_size, stride, dilation=None):
    """According to https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    if dilation is None:
        dilation = (1, ) * 2
    if isinstance(padding, int):
        padding = (padding, ) * 2
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, ) * 2
    if isinstance(stride, int):
        stride = (stride, ) * 2

    output_size = (
        out_channels,
        np.floor((input_size[1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) /
                 stride[0] + 1).astype(int),
        np.floor((input_size[2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) /
                 stride[1] + 1).astype(int)
    )
    return output_size


c_i, c_o = 3, 16
k, s, p = 3, 2, 1


sample_2d_tensor = torch.ones((c_i, 64, 64))
c2d = nn.Conv2d(in_channels=c_i, out_channels=c_o, kernel_size=k,
                stride=s, padding=p)

output_size = conv2d_output_size(
    sample_2d_tensor.shape, out_channels=c_o, kernel_size=k, stride=s, padding=p)

print("After conv2d")
print("Dummy input size:", sample_2d_tensor.shape)
print("Calculated output size:", output_size)
print("Real output size:", c2d(sample_2d_tensor).detach().numpy().shape)

val = 1
for x in output_size:
  val = val * x
