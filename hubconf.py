import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torchmetrics import Precision, Recall, F1Score, Accuracy
from torchmetrics.classification import accuracy


device="cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Define model
#---------------------------------------------------------------------------------------
class cs21m009(nn.Module):
    def __init__(self,num_channels,num_classes,h,w,config):
        super(cs21m009,self).__init__()
        
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
#---------------------------------------------------------------------------------------

def get_lossfn_and_optimizer():
    lossfn = loss_fn
    
    return lossfn

# cross entropy 
def loss_fn(y_pred,y_true):
    e=0.0001
    v=-torch.sum(y_true*torch.log(y_pred+e))
    return v
#-------------------------------------------------------------------------------
def load_data():
    training_data = datasets.FashionMNIST(
                root="data",
                train = True,
                download = True,
                transform = ToTensor())
    
    test_data = datasets.FashionMNIST(
                root="data",
                train=False,
                download=True,
                transform = ToTensor())
    
    return training_data, test_data
#-------------------------------------------------------------------------------
def create_dataloaders(training_data, test_data, batch_size=64):

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
        
    return train_dataloader, test_dataloader
#-----------------------------------------------------------------
def get_model(trainloader,config,epochs,learning_rate):
    
    N , num_channels , height , width = next(iter(trainloader))[0].shape
    num_classes = len(trainloader.dataset.classes)
    
    my_model = cs21m009(num_channels,num_classes,height,width,config).to(device)
    
    train(trainloader,my_model,epochs,learning_rate)
    
    return my_model
#-----------------------------------------------------------------

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

#--------------------------------------------------------------------



def test(dataloader,model,loss_fn):
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
#--------------------------------------------------------------------------------------------------
