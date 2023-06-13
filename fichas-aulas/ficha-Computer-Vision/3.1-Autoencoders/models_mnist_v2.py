# modelos de autoencoders para o mnist 
import torch
import torch.nn as nn
from torch.nn import Sequential
from torch.nn import Linear
from torch.nn import Flatten
from torch.nn import BatchNorm1d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import BatchNorm2d
from torch.nn import ConvTranspose2d
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Dropout
from torch.nn import Softmax, Tanh
from torch.nn import Module
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_


# Definição classe para o Autoencoder MLP default: 784-512-256-2-256-512-784
class autoencoderMLP(Module):
    def __init__(self, x_dim=28*28, h_dim1=512, h_dim2=256, ls_dim=2):
        super(autoencoderMLP, self).__init__()
        # encoder
        self.fc1 = Linear(x_dim, h_dim1)
        self.fc2 = Linear(h_dim1, h_dim2)
        self.fc3 = Linear(h_dim2, ls_dim)
        # decoder
        self.fc4 = Linear(ls_dim, h_dim2)
        self.fc5 = Linear(h_dim2, h_dim1)
        self.fc6 = Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        x = x.relu()
        x = self.fc3(x)
        return x
        
    def decoder(self, z):
        z = self.fc4(z)
        z = z.relu()
        z = self.fc5(z)
        z = z.relu()
        z = self.fc6(z)
        z = z.sigmoid()
        return z
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.encoder(x)
        z = self.decoder(x)
        return z

#modelo treinado com a seguinte configuração:
#model = models_mnist.AE_MLP_P(x_dim=28*28, h_dim1=256, h_dim2=128, h_dim3=64, ls_dim=2)
#gravado no ficheiro: model= torch.load('AE_MLP_P_MNIST.pth')
class AE_MLP_P(Module):
    def __init__(self, x_dim=28*28, h_dim1=512, h_dim2=256, h_dim3=128, ls_dim=2):
        super(AE_MLP_P, self).__init__()
        # encoder
        self.fc1 = Linear(x_dim, h_dim1)
        self.fc2 = Linear(h_dim1, h_dim2)
        self.fc3 = Linear(h_dim2, h_dim3)
        self.fc4 = Linear(h_dim3, ls_dim)
        # decoder
        self.fc5 = Linear(ls_dim, h_dim3)
        self.fc6 = Linear(h_dim3, h_dim2)
        self.fc7 = Linear(h_dim2, h_dim1)
        self.fc8 = Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x).relu()
        x = self.fc4(x)
        return x
        
    def decoder(self, z):
        z = self.fc5(z).relu()
        z = self.fc6(z).relu()
        z = self.fc7(z).relu()
        z = self.fc8(z).sigmoid()
        return z
    
    def forward(self, x):
        x = x.view(-1, 784)
        ls = self.encoder(x)
        z = self.decoder(ls)
        return z, ls

    
    # Definição classe para o Autoencoder MLP default: 784-512-256-2-256-512-784
class autoencoderMLP_BN(Module):
    def __init__(self, x_dim=28*28, h_dim1=512, h_dim2=256, ls_dim=2):
        super(autoencoderMLP_BN, self).__init__()
        # encoder
        self.fc1 = Linear(x_dim, h_dim1)
        self.fc2 = Linear(h_dim1, h_dim2)
        self.fc3 = Linear(h_dim2, ls_dim)
        # decoder
        self.fc4 = Linear(ls_dim, h_dim2)
        self.fc5 = Linear(h_dim2, h_dim1)
        self.fc6 = Linear(h_dim1, x_dim)
        self.bn1 = BatchNorm1d(h_dim1)
        self.bn2 = BatchNorm1d(h_dim2)
  
    def encoder(self, x):
        x = self.fc1(x)
        x = x.relu()
        x = self.bn1(x)
        x = self.fc2(x)
        x = x.relu()
        x = self.bn2(x)
        x = self.fc3(x)
        return x

    def decoder(self, z):
        z = self.fc4(z)
        z = z.relu()
        z = self.bn2(z)
        z = self.fc5(z)
        z = z.relu()
        z = self.bn1(z)
        z = self.fc6(z)
        z = z.sigmoid()
        return z
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.encoder(x)
        z = self.decoder(x)
        return z

class autoencoderMLP_DROP(Module):
    def __init__(self, x_dim=28*28, h_dim1=512, h_dim2=256, ls_dim=2, drop=0.5):
        super(autoencoderMLP_DROP, self).__init__()
        # encoder
        self.fc1 = Linear(x_dim, h_dim1)
        self.fc2 = Linear(h_dim1, h_dim2)
        self.fc3 = Linear(h_dim2, ls_dim)
        # decoder
        self.fc4 = Linear(ls_dim, h_dim2)
        self.fc5 = Linear(h_dim2, h_dim1)
        self.fc6 = Linear(h_dim1, x_dim)
        self.bn1 = BatchNorm1d(h_dim1)
        self.bn2 = BatchNorm1d(h_dim2)
        self.drop = Dropout(drop)
  
    def encoder(self, x):
        x = self.fc1(x)
        x = x.relu()
        x = self.drop(x)
        x = self.fc2(x)
        x = x.relu()
        x = self.drop(x)
        x = self.fc3(x)
        return x

    def decoder(self, z):
        z = self.fc4(z)
        z = z.relu()
        z = self.drop(z)
        z = self.fc5(z)
        z = z.relu()
        z = self.drop(z)
        z = self.fc6(z)
        z = z.sigmoid()
        return z
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.encoder(x)
        z = self.decoder(x)
        return z
    
    
    
    
class autoencoderCONV(Module):
    def __init__(self):
        super(autoencoderCONV, self).__init__()
        self.encoder = Sequential(Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
                                  ReLU(True),
                                  MaxPool2d(2, stride=2),  # b, 16, 5, 5
                                  Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
                                  ReLU(True),
                                  MaxPool2d(2, stride=1))  # b, 8, 2, 2
        self.decoder = Sequential(ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
                                  ReLU(True),
                                  ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
                                  ReLU(True),
                                  ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
                                  Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x    
    
#modelo treinado com a seguinte configuração:
#model = AE_CONV()
#gravado no ficheiro: model= torch.load('AE_CONV_MNIST.pth')  
class AE_CONV(Module):
    def __init__(self, final=True):
        super(AE_CONV,self).__init__()
        self.encoder=Sequential(
                Conv2d(1,128,kernel_size=7),#32*22*22
                BatchNorm2d(128),
                ReLU(),
                MaxPool2d(kernel_size=2,stride=2),#32*11*11
                Conv2d(128,64,kernel_size=3),#8*9*9
                BatchNorm2d(64),
                ReLU(),
                MaxPool2d(kernel_size=2,stride=2),#8*4*4
                Conv2d(64,1,kernel_size=2),#2*3*3
                BatchNorm2d(1),
                ReLU(),
            )
        self.decoder=Sequential(
                ConvTranspose2d(1,64,kernel_size=5,stride=2),#8*9*9
                BatchNorm2d(64),
                ReLU(),
                ConvTranspose2d(64,128,kernel_size=7,stride=2),#4*23*23
                BatchNorm2d(128),
                ReLU(),
                ConvTranspose2d(128,1,kernel_size=6,stride=1),#1*28*28
                BatchNorm2d(1),
                Sigmoid(),
            )
        
    def forward(self,x):
        ls=self.encoder(x)
        z=self.decoder(ls)
        return z,ls

#modelo treinado com a seguinte configuração:
#model = VAE_MLP(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2)
#gravado no ficheiro: model= torch.load('VAE_MLP_MNIST.pth')
class VAE_MLP(Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE_MLP, self).__init__()
        self.x_dim=x_dim
        # encoder
        self.fc1 = Linear(x_dim, h_dim1)
        self.fc2 = Linear(h_dim1, h_dim2)
        self.fc_mu = Linear(h_dim2, z_dim)
        self.fc_log_var = Linear(h_dim2, z_dim)
        # decoder
        self.fc4 = Linear(z_dim, h_dim2)
        self.fc5 = Linear(h_dim2, h_dim1)
        self.fc6 = Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = self.fc1(x).relu()
        h = self.fc2(h).relu()
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = eps.mul(std).add_(mu)
        return sample
        
    def decoder(self, z):
        z = self.fc4(z).relu()
        z = self.fc5(z).relu()
        z = self.fc6(z).sigmoid()
        return z 
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, self.x_dim))
        sample = self.sampling(mu, log_var)
        outputs = self.decoder(sample)
        return outputs, mu, log_var, sample


#modelo treinado com a seguinte configuração:
#model = VAE_CONV()
#gravado no ficheiro: model= torch.load('VAE_CONV_MNIST.pth')
class VAE_CONV(Module):
    def __init__(self, final=True):
        super(VAE_CONV,self).__init__()
        #encoder
        self.conv1 = Conv2d(1,32,kernel_size=3,padding=1) #32*22*22
        #self.pool1 = MaxPool2d(kernel_size=2,stride=2) #32*11*11
        self.conv2 = Conv2d(32,64,kernel_size=3,stride=2,padding=1) #8*9*9
        #self.pool2 = MaxPool2d(kernel_size=2,stride=2) #8*4*4
        self.conv3 = Conv2d(64,64,kernel_size=3,padding=1) #4*3*3
        self.conv4 = Conv2d(64,64,kernel_size=3,padding=1) #4*3*3
        self.fc_mu = Linear(12544,32)
        self.fc_log_var = Linear(12544,32)
        #decoder
        self.fc4 = Linear(32, 12544) 
        self.convT1 = ConvTranspose2d(64,32,kernel_size=3,stride=2) #8*9*9
        self.conv5 = Conv2d(32,1,kernel_size=2)
        #self.convT2 = ConvTranspose2d(8,4,kernel_size=7,stride=2) #4*23*23
        #self.convT4 = ConvTranspose2d(4,1,kernel_size=6,stride=1)#1*28*28     
 
    def encoder(self, x):
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).relu()
        x = self.conv4(x).relu()
        x = torch.flatten(x, start_dim=1)
        #x = self.fc1(x).relu()
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var
    
    def decoder(self, z):
        #z = self.fc3(z).relu()
        z = self.fc4(z).relu()
        z= z.view(-1,64,14,14)
        z = self.convT1(z).relu()
        #z = self.convT4(z) 
        z = self.conv5(z).sigmoid()
        return z

    def sampling(self, mu, log_var):
        #param mu: mean from the encoder's latent space
        #param log_var: log variance from the encoder's latent space
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        #sample = eps.mul(std).add_(mu) # alternativa
        return sample
    
    def forward(self,x,final=True):
        mu, log_var =self.encoder(x)
        sample=self.sampling(mu, log_var)
        outputs=self.decoder(sample)
        return outputs, mu, log_var, sample


#modelo treinado com a seguinte configuração:
#model = CVAE_MLP(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2)
#gravado no ficheiro: model= torch.load('CVAE_MLP_MNIST.pth')
class CVAE_MLP(Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, c_dim):
        super(CVAE_MLP, self).__init__()
        # encoder
        self.fc1 = Linear(x_dim + c_dim, h_dim1)
        self.fc2 = Linear(h_dim1, h_dim2)
        self.fc_mu = Linear(h_dim2, z_dim)
        self.fc_log_var = Linear(h_dim2, z_dim)
        # decoder
        self.fc4 = Linear(z_dim + c_dim, h_dim2)
        self.fc5 = Linear(h_dim2, h_dim1)
        self.fc6 = Linear(h_dim1, x_dim)
        
    def encoder(self, x, c):
        concat_input = torch.cat([x, c], 1)
        h = self.fc1(concat_input).relu()
        h = self.fc2(h).relu()
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = eps.mul(std).add_(mu)
        return sample
        
    def decoder(self, z, c):
        concat_input = torch.cat([z, c], 1)
        z = self.fc4(concat_input).relu()
        z = self.fc5(z).relu()
        z = self.fc6(z).sigmoid()
        return z 
    
    def forward(self, x, c):
        mu, log_var = self.encoder(x.view(-1, 784), c)
        sample = self.sampling(mu, log_var)
        outputs = self.decoder(sample, c)
        return outputs, mu, log_var, sample






'''
kernel_size = 4 # (4, 4) kernel
init_channels = 8 # initial number of filters
image_channels = 1 # MNIST images are grayscale
latent_dim = 16 # latent dimension for sampling
'''
# define a Conv VAE
class ConvVAE(Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
 
        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels*4, out_channels=64, kernel_size=kernel_size, 
            stride=2, padding=0
        )
        # fully connected layers for learning representations
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)
        # decoder 
        self.dec1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=init_channels*8, kernel_size=kernel_size, 
            stride=1, padding=0
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels*2, out_channels=image_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        z = z.view(-1, 64, 1, 1)
 
        # decoding
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        reconstruction = torch.sigmoid(self.dec4(x))
        return reconstruction, mu, log_var