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


