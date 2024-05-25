import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

  
from collections import OrderedDict




LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
 
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

  



class ConvBlock(nn.Module):
   
    def __init__(self, in_channels: int, 
                        out_channels: int,
                        kernel_size: int, 
                        stride: int = 2, 
                        padding: int = 1, 
                        slope: float = 0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=slope)
    
    def forward(self, x):
  
        return self.relu(self.bn(self.conv(x)))

 

class Encoder(nn.Module):
 
    def __init__(self, z_dim: int = 32):
        super().__init__()

        # encoder
        self.encoder = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 32, 4, stride=2, padding=1)),
            ("relu1", nn.LeakyReLU(0.2)), #32x32x32
            ("block1", ConvBlock(32, 64, 4, stride=2, padding=1, slope=0.2)), # 64x16x16
            ("block2", ConvBlock(64, 128, 4, stride=2, padding=1, slope=0.2)), # 128x8x8
            ("block3", ConvBlock(128, 256, 4, stride=2, padding=1, slope=0.2)), # 256x4x4
        ]))

        self.fc = nn.Linear(4096, z_dim)
      
 
    def forward(self, x):
 
        x = self.encoder(x)
        x = x.view(-1, 4096)
        return self.fc(x)






class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super().__init__()
 
        self.enc1 = Encoder(num_inputs)
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
 
        self.apply(weights_init_)


    def forward(self, state, action):
         
        x1 = torch.cat([self.enc1(state), action], 1)        
        x1 = F.relu(self.linear1(x1))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
 
        return x1


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super().__init__()
        
        self.enc = Encoder(num_inputs)
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim) # 32, 512.
        self.linear2 = nn.Linear(hidden_dim, hidden_dim) # 512, 512.

        self.mean_linear = nn.Linear(hidden_dim, num_actions) # 512, 2.
        self.log_std_linear = nn.Linear(hidden_dim, num_actions) # 512, 2.

        self.apply(weights_init_)


    def forward(self, state, vel):
        
        # state = self.enc(state)
        x = torch.cat([self.enc(state), vel], 1)        

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std


    def sample(self, state, vel):
        mean, log_std = self.forward(state, vel)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, torch.tanh(mean) # (b, act_dim), (b, 1), (b, act_dim). 


 















