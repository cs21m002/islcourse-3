import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import numpy as np

class my_model(nn.Module):
    def __init__(self,size):
        super(my_model,self).__init__()
        self.net_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20,kernel_size=5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(size,10),
            nn.Softmax()
        )
    
    def forward(self,x):
        x = self.net_stack(x)
        return x
      



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
        
def get_model(train_loader):

  c_i, c_o = 3, 20
  k, s, p = 5, 2, 1

  for x,y in train_loader:
    x_shape = x.shape
    break

  print(x_shape)

  c_i = x_shape[1]

  sample_2d_tensor = torch.ones((x_shape[1], x_shape[2], x_shape[3]))
  c2d = nn.Conv2d(in_channels=c_i, out_channels=c_o , kernel_size=k)

  output_size = conv2d_output_size(
      sample_2d_tensor.shape, out_channels=c_o , kernel_size= k , stride = 1 , padding = 0)

  print("After conv2d")
  print("Dummy input size:", sample_2d_tensor.shape)
  print("Calculated output size:", output_size)
  print("Real output size:", c2d(sample_2d_tensor).detach().numpy().shape)

  val = 1
  for x in output_size:
    val = val * x
  
  print (val)
  
  model = my_model(val)

  return model


