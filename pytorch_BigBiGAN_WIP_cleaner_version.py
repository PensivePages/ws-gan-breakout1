from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

#base code from AISC GAN Workshop

#--Choose GAN type--
DCGAN = False #change to 'False' for BigBiGAN
if not DCGAN:
  BigBiGAN = True #using elif for legability and ctrl + f searchability
else:
  BigBiGAN = False

# Set random seem for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot = "./ImageNet/content/train"#/ships" #'/content/drive/My Drive/gans_workshop/fonts/sheets'
#cutting off '/ships' seems to have solved the "no files found error, wierd... but explained by this post:
#https://stackoverflow.com/questions/54613573/runtimeerror-found-0-files-in-subfolders-of-error-about-subfolder-in-pytor

# Number of workers for dataloader
workers = 4#0#4
#setting workers to 0 seems to have fixed the "RuntimeError: DataLoader worker (pid(s) [bla bla]) exited unexpectedly"
# fix from here: https://github.com/pytorch/pytorch/issues/5301

# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
if DCGAN:
  nz = 100
else: #BigBiGAN
  nz = 196

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 15#5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Size of feature maps in Linear discriminators
#channels = # may use this later when I switch to dims * dims

# H (unary: E(x), z) Linear discriminator dims
h_input_size = 196#12800 #--!!--NOTICE-- this is the flattening of the concatenated latent spaces - should really be coded to dims * dims    
h_hidden_size = 196#6560 # reduce dim in the directions of F's flattened output size | 12800 + 320 / 2 = 6560
h_output_size = 196#320 # F's flattened output size

# J joint (xz) Linear Discriminator dims | 320 is just transforming a few times with same dims for final add (x + xz + z)
j_input_size = 392#320
j_hidden_size = 392#320 
j_output_size = 392#320

# F score S_x(F(X))
Sx_input_size = 196
Sx_hidden_size = 98
Sx_output_size = 1

# H score S_z(H(Z))
Sz_input_size = 196
Sz_hidden_size = 98
Sz_output_size = 1

# J score S_xz(J(F(X), H(Z)))
Sxz_input_size = 392
Sxz_hidden_size = 196
Sxz_output_size = 1

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, init_zero_weights=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, bias=False):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        # Fill this in
        self.convT1 = deconv(nz, ngf*8, 4, 1, 0)
        self.convT2 = deconv(ngf*8, ngf*4, 4, 2, 1)
        self.convT3 = deconv(ngf*4, ngf*2, 4, 2, 1)
        self.convT4 = deconv(ngf*2, ngf, 4, 2, 1)
        self.output = deconv(ngf, nc, 4, 2, 1, batch_norm=False)

    def forward(self, input):
        out = F.relu(self.convT1(input), inplace=True)
        out = F.relu(self.convT2(out), inplace=True)
        out = F.relu(self.convT3(out), inplace=True)
        out = F.relu(self.convT4(out), inplace=True)

        # What non-linearity on the output layer
        #return ?(self.output(out))
        return F.tanh(self.output(out))

# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)

#--Encoder (WIP)--

class Encoder(nn.Module):
    def __init__(self, ngpu):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.conv1 = conv(nc, ndf, 4, 2, 1)
        self.conv2 = conv(ndf, ndf * 2, 4, 2, 1)
        self.conv3 = conv(ndf * 2, ndf * 4, 4, 2, 1)
        self.conv4 = conv(ndf * 4, ndf * 8, 4, 2, 1)
        self.output = conv(ndf * 8, nz, 4, 1, 0, batch_norm=False)
        #noise = torch.randn(b_size, nz, 1, 1, device=device) #(just here for my own reference) b_size looks like it's 'batch_size which is 64'

    def forward(self, input):
        out = F.leaky_relu(self.conv1(input), inplace=True)
        out = F.leaky_relu(self.conv2(out), inplace=True)
        out = F.leaky_relu(self.conv3(out), inplace=True)
        out = F.leaky_relu(self.conv4(out), inplace=True)
        
        # what does the return line look like?
        return F.tanh(self.output(out))#.view(batch_size, nz, 1, 1) #put tanh here instead of sigmoid, but really... should the activation be left out or maybe changed to CReLU?


# Create the Discriminator
netE = Encoder(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netE = nn.DataParallel(netE, list(range(ngpu)))
    
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netE.apply(weights_init)

# Print the model
print(netE)

if DCGAN:
  class F_Discriminator(nn.Module):
      def __init__(self, ngpu, DCGAN):
          super(F_Discriminator, self).__init__()
          self.ngpu = ngpu
          # Fill this in
          self.conv1 = conv(nc, ndf, 4, 2, 1) #changing strides to 1 and keral to 3 for larger out
          self.conv2 = conv(ndf, ndf * 2, 4, 2, 1)
          self.conv3 = conv(ndf * 2, ndf * 4, 4, 2, 1)
          self.conv4 = conv(ndf * 4, ndf * 8, 4, 2, 1)
          self.output = conv(ndf * 8, 1, 4, 1, 0, batch_norm=False)

      def forward(self, input):
          out = F.leaky_relu(self.conv1(input), inplace=True)
          out = F.leaky_relu(self.conv2(out), inplace=True)
          out = F.leaky_relu(self.conv3(out), inplace=True)
          out = F.leaky_relu(self.conv4(out), inplace=True)

          if DCGAN:
            return F.sigmoid(self.output(out))
          else:
            return F.leaky_relu(self.output(out)) #changed to F.leaky_relu because more appropriate for how layer used as 'F'
else: #BigBiGAN
  class F_Discriminator(nn.Module):
      def __init__(self, ngpu, DCGAN):
          super(F_Discriminator, self).__init__()
          self.ngpu = ngpu
          # Fill this in
          self.conv1 = conv(nc, ndf, 3, 2, 1) #changing strides to 1 and keral to 3 for larger out
          self.conv2 = conv(ndf, ndf * 2, 3, 2, 1)
          self.conv3 = conv(ndf * 2, ndf * 4, 3, 1, 1)
          self.conv4 = conv(ndf * 4, ndf * 8, 3, 1, 1)
          self.output = conv(ndf * 8, 1, 3, 1, 0, batch_norm=False)

      def forward(self, input):
          out = F.leaky_relu(self.conv1(input), inplace=True)
          out = F.leaky_relu(self.conv2(out), inplace=True)
          out = F.leaky_relu(self.conv3(out), inplace=True)
          out = F.leaky_relu(self.conv4(out), inplace=True)

          if DCGAN:
            return F.sigmoid(self.output(out))
          else:
            return F.leaky_relu(self.output(out)) #changed to F.leaky_relu because more appropriate for how layer used as 'F'

# Create the Discriminator
netD = F_Discriminator(ngpu, DCGAN).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
    
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)

class F_Score(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, ngpu):
        super(F_Score, self).__init__()
        self.nonLinearity = torch.nn.LeakyReLU()
#         self.input = nn.Linear(input_size, hidden_size)
#         self.hidden = nn.Linear(hidden_size, hidden_size)
#         self.output = nn.Linear(hidden_size, output_size)
        self.output = nn.Linear(input_size, output_size)


    def forward(self, x):
#         x = self.nonLinearity(self.input(x))
#         x = self.nonLinearity( self.hidden(x) )
        #return self.nonLinearity(self.output(x))
        #return F.sigmoid(self.output(x))
        return self.output(x) #returing raw out because sig doesn't seem to make sense to me when score is used with hinge


# Create the Discriminator
scoreF = F_Score(input_size=Sx_input_size, hidden_size=Sx_hidden_size, output_size=Sx_output_size, ngpu=ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    scoreF = nn.DataParallel(scoreF, list(range(ngpu)))
    
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
scoreF.apply(weights_init)

# Print the model
print(scoreF)

class H_Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, ngpu):
        super(H_Discriminator, self).__init__()
        self.nonLinearity = torch.nn.LeakyReLU()
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x = self.nonLinearity(self.input(x))
        x = self.nonLinearity( self.hidden(x) )
        return self.nonLinearity(self.output(x))


# Create the Discriminator
netH = H_Discriminator(input_size=h_input_size, hidden_size=h_hidden_size, output_size=h_output_size, ngpu=ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netH = nn.DataParallel(netH, list(range(ngpu)))
    
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netH.apply(weights_init)

# Print the model
print(netH)

class H_Score(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, ngpu):
        super(H_Score, self).__init__()
        self.nonLinearity = torch.nn.LeakyReLU()
#         self.input = nn.Linear(input_size, hidden_size)
#         self.hidden = nn.Linear(hidden_size, hidden_size)
#         self.output = nn.Linear(hidden_size, output_size)
        self.output = nn.Linear(input_size, output_size)


    def forward(self, x):
#         x = self.nonLinearity(self.input(x))
#         x = self.nonLinearity( self.hidden(x) )
        #return self.nonLinearity(self.output(x))
        #return F.sigmoid(self.output(x))
        return self.output(x) #returing raw out because sig doesn't seem to make sense to me when score is used with hinge


# Create the Discriminator
scoreH = H_Score(input_size=Sz_input_size, hidden_size=Sz_hidden_size, output_size=Sz_output_size, ngpu=ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    scoreH = nn.DataParallel(scoreH, list(range(ngpu)))
    
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
scoreH.apply(weights_init)

# Print the model
print(scoreH)

class J_Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, ngpu):
        super(J_Discriminator, self).__init__()
        self.nonLinearity = torch.nn.LeakyReLU()
        #self.nonLinearity = torch.nn.ReLU()
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x = self.nonLinearity(self.input(x))
        x = self.nonLinearity( self.hidden(x) )
        return self.nonLinearity(self.output(x))
        #return F.sigmoid(self.output(x)) #going back to ReLU atm, because of use of hinge loss


# Create the Discriminator
netJ = J_Discriminator(input_size=j_input_size, hidden_size=j_hidden_size, output_size=j_output_size, ngpu=ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netJ = nn.DataParallel(netJ, list(range(ngpu)))
    
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netJ.apply(weights_init)

# Print the model
print(netJ)

class J_Score(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, ngpu):
        super(J_Score, self).__init__()
        self.nonLinearity = torch.nn.LeakyReLU()
#         self.input = nn.Linear(input_size, hidden_size)
#         self.hidden = nn.Linear(hidden_size, hidden_size)
#         self.output = nn.Linear(hidden_size, output_size)
        self.output = nn.Linear(input_size, output_size)


    def forward(self, x):
#         x = self.nonLinearity(self.input(x))
#         x = self.nonLinearity( self.hidden(x) )
        #return self.nonLinearity(self.output(x))
        #return F.sigmoid(self.output(x))
        return self.output(x) #returing raw out because sig doesn't seem to make sense to me when score is used with hinge


# Create the Discriminator
scoreJ = J_Score(input_size=Sxz_input_size, hidden_size=Sxz_hidden_size, output_size=Sxz_output_size, ngpu=ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    scoreJ = nn.DataParallel(scoreJ, list(range(ngpu)))
    
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
scoreJ.apply(weights_init)

# Print the model
print(scoreJ)

# Hinge Loss (from: https://github.com/ajbrock/BigGAN-PyTorch/blob/master/losses.py)
#Note: not in use
def loss_hinge_dis(dis_real, dis_fake): # reversed input cuz THATS WHY! :-P
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake
# def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
  # loss = torch.mean(F.relu(1. - dis_real))
  # loss += torch.mean(F.relu(1. + dis_fake))
  # return loss

#Note: not in use  
def loss_hinge_gen(dis_fake):
  loss = -torch.mean(dis_fake)
  return loss

#Note: not in use
def loss_hinge_enc(dis_real): #added
  loss = torch.mean(dis_real)
  return loss

# Initialize BCELoss function (bynary cross entropy loss)
criterion = nn.BCELoss()
if DCGAN:
  criterion = nn.BCELoss()
else: #BigBiGAN
  #criterion = loss_hinge_dis(*args) moved to BigBiGAN section atm
  pass

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerE = optim.Adam(netE.parameters(), lr=lr, betas=(beta1, 0.999)) #added
optimizerH = optim.Adam(netH.parameters(), lr=lr, betas=(beta1, 0.999)) #added
optimizerJ = optim.Adam(netJ.parameters(), lr=lr, betas=(beta1, 0.999)) #added
optimizerSx = optim.Adam(scoreF.parameters(), lr=lr, betas=(beta1, 0.999)) #added
optimizerSz = optim.Adam(scoreH.parameters(), lr=lr, betas=(beta1, 0.999)) #added
optimizerSxz = optim.Adam(scoreJ.parameters(), lr=lr, betas=(beta1, 0.999)) #added

checkpoint_dir = '/content/drive/My Drive/ImageNet/Training/'

def create_dir(directory):
    """Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

create_dir(checkpoint_dir)
def checkpoint(iteration, G, D, E, H, J, Sx, Sxz, Sz):
    """
    Saves the parameters of the generator G and discriminator D.
    """
    G_path = os.path.join(checkpoint_dir, 'G.pkl')
    D_path = os.path.join(checkpoint_dir, 'D.pkl') # F
    E_path = os.path.join(checkpoint_dir, 'E.pkl')
    H_path = os.path.join(checkpoint_dir, 'H.pkl')
    J_path = os.path.join(checkpoint_dir, 'J.pkl')
    Sx_path = os.path.join(checkpoint_dir, 'Sx.pkl')
    Sxz_path = os.path.join(checkpoint_dir, 'Sxz.pkl')
    Sz_path = os.path.join(checkpoint_dir, 'Sz.pkl')
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path) # F
    torch.save(E.state_dict(), E_path)
    torch.save(H.state_dict(), H_path)
    torch.save(J.state_dict(), J_path)
    torch.save(Sx.state_dict(), Sx_path)
    torch.save(Sxz.state_dict(), Sxz_path)
    torch.save(Sz.state_dict(), Sz_path)

def load_checkpoint(model, checkpoint_name):
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, checkpoint_name)))
    
    
load = False
if load: # NOTICE: (untested)
  load_checkpoint(netG, 'G.pkl')
  load_checkpoint(netE, 'E.pkl')
  load_checkpoint(netD, 'F.pkl') # F
  load_checkpoint(netH, 'H.pkl')
  load_checkpoint(netJ, 'J.pkl')
  load_checkpoint(scoreF, 'Sx.pkl')
  load_checkpoint(scoreH, 'Sxz.pkl')
  load_checkpoint(scoreJ, 'Sz.pkl')


    # Training Loop 

# Lists to keep track of progress
img_list = []
E_img_list = [] #empty atm, need to keep and collect encoder outs
# ^--!!!--Notice: instead of showing E(x) (to see aht it looks like, because I'm currious :-P), I probably want to do G(E(x))
G_losses = []
D_losses = []
E_losses = [] #added
iters = 0
print_100 = True
#print_i = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs*25):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        
        if DCGAN:
        
          ############################
          # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
          ###########################
          ## Train with all-real batch
          netD.zero_grad()
          # Format batch
          real_cpu = data[0].to(device)
          b_size = real_cpu.size(0)
          label = torch.full((b_size,), real_label, device=device)
          # Forward pass real batch through D
          output = netD(real_cpu).view(-1)
          # Calculate loss on all-real batch
          errD_real = criterion(output, label)
          # Calculate gradients for D in backward pass
          errD_real.backward()
          D_x = output.mean().item()

          ## Train with all-fake batch
          # Generate batch of latent vectors
          noise = torch.randn(b_size, nz, 1, 1, device=device)
          # Generate fake image batch with G
          fake = netG(noise)
          label.fill_(fake_label)
          # Classify all fake batch with D
          output = netD(fake.detach()).view(-1)
          # Calculate D's loss on the all-fake batch
          errD_fake = criterion(output, label)
          # Calculate the gradients for this batch
          errD_fake.backward()
          D_G_z1 = output.mean().item()
          # Add the gradients from the all-real and all-fake batches
          errD = errD_real + errD_fake
          # Update D
          optimizerD.step()

          ############################
          # (2) Update G network: maximize log(D(G(z)))
          ###########################
          netG.zero_grad()
          label.fill_(real_label)  # fake labels are real for generator cost
          # Since we just updated D, perform another forward pass of all-fake batch through D
          output = netD(fake).view(-1)
          # Calculate G's loss based on this output
          errG = criterion(output, label)
          # Calculate gradients for G
          errG.backward()
          D_G_z2 = output.mean().item()
          # Update G
          optimizerG.step()

          # Output training stats
          if i % 50 == 0:
              print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                       errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

          # Save Losses for plotting later
          G_losses.append(errG.item())
          D_losses.append(errD.item())

#           # Check how the generator is doing by saving G's output on fixed_noise
#           if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
#               with torch.no_grad():
#                   fake = netG(fixed_noise).detach().cpu()
#               img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

              
          # Check how the generator is doing by saving G's output on fixed_noise
          if (iters % 50 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
              with torch.no_grad():
                  fake = netG(fixed_noise).detach().cpu()
              img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
              if print_100:
                # Plot the fake images from this epoch
                plt.subplot(1,2,2)
                plt.axis("off")
                plt.title("Fake Images")
                plt.imshow(np.transpose(img_list[-1],(1,2,0)))
                plt.show()
        
        elif BigBiGAN: #used elif for legability and ctrl + f searchability
          #print("BigBiGAN is Under construction")
          #break
          ############################
          # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
          ###########################
          
          #---"Autoencoder" Section | enc, gen
          
          #--run the encoder--
          ## Training components x -> E(x) batch
          real_cpu = data[0].to(device)
          b_size = real_cpu.size(0)
          E_output = netE(real_cpu).view(b_size, -1)

          #--run the generator--
          ## Training components z -> G(z) batch
          noise = torch.randn(b_size, nz, 1, 1, device=device)
          G_output = fake = netG(noise)
          
          #---Unary Discriminators Section | F (which is 'D' here), H
          
          ## Train with all-real batch
          #--run F on X (ie: X = (x and G(z)))-- Discriminates between encoder (real) and generator (fake)
          netD.zero_grad()
          F_output_real = netD(real_cpu).view(b_size, -1)
          Sx_real = scoreF(F_output_real)
          F_output_fake = netD(G_output).view(b_size, -1)
          Sx_fake = scoreF(F_output_fake)

          ## Train with all-fake batch
          #--run H on Z (ie: Z = (E(x) and z))-- Discriminates between encoder (real) and generator (fake)
          netH.zero_grad()
          H_output_real = netH(E_output.detach()).view(b_size, -1)
          Sz_real = scoreH(H_output_real)
          noise = noise.view(b_size, -1) #reshape for H
          H_output_fake = netH(noise.detach()).view(b_size, -1)
          Sz_fake = scoreH(H_output_fake)
          
          #---Joint (xz) Discriminator | J
          xz_input_real = torch.cat((F_output_real, H_output_real), -1)
          xz_input_fake = torch.cat((F_output_fake, H_output_fake), -1)
          
          #--run J on encoder terms (real) (ie: X = (F(x), H(E(x)))-- Discriminates between encoder (real) and generator (fake)
          netJ.zero_grad()
          xz_real_cpu = xz_input_real.to(device)
          b_size = xz_real_cpu.size(0)
          J_output_real = netJ(xz_real_cpu)#.view(b_size, -1)
          Sxz_real = scoreJ(J_output_real)
          
          #--run J on ganerator terms (fake) (ie: Z = (F(G(z)), H(z)))-- Discriminates between encoder (real) and generator (fake)
          #print("z_output.size(): ", z_output.size())
          J_output_fake = netJ(xz_input_fake.detach())#.view(b_size, -1)
          Sxz_fake = scoreJ(J_output_fake)
          
          #--Loss_D  = hinge(y(Sx)) + hinge(y(Sxz)) + hinge(y(Sz)) where y in {-1, 1}
          #Loss_D_real = F.relu(2. - Sx_real) + F.relu(2. - Sxz_real) + F.relu(2. - Sz_real) + .000001#).view(-1) #havent tested, leads to a "can't interpolate non-scalar error," or something similar
          #Loss_D_fake = F.relu(2. + Sx_fake) + F.relu(2. + Sxz_fake) + F.relu(2. + Sz_fake) + .000001#).view(-1) #havent tested, leads to a "can't interpolate non-scalar error," or something similar
          Loss_D_real = F.sigmoid(F.relu(2. - Sx_real) + F.relu(2. - Sxz_real) + F.relu(2. - Sz_real) + .000001).view(-1)
          Loss_D_fake = F.sigmoid(F.relu(2. + Sx_fake) + F.relu(2. + Sxz_fake) + F.relu(2. + Sz_fake) + .000001).view(-1)
          label = torch.full((b_size,), real_label, device=device)
          Loss_D_real = criterion(Loss_D_real, label)
          Loss_D_real.backward(retain_graph=True)
          label.fill_(fake_label)
          Loss_D_fake = criterion(Loss_D_fake, label)
          Loss_D_fake.backward(retain_graph=True)
          errXZ = Loss_D_real + Loss_D_fake
          # Update Sxz, Sz, Sx, J, H, F
          optimizerSxz.step()
          optimizerSz.step()
          optimizerSx.step()
          optimizerJ.step()
          optimizerH.step()
          optimizerD.step() #D is F

          #--updating generator--
          netG.zero_grad()
          label.fill_(real_label)  #trick 
          #label.fill_(fake_label) #actual
          outputF = netD(G_output).view(b_size, -1)
          outputH = netH(noise).view(b_size, -1)
          xz_input = torch.cat((outputF, outputH), -1)
          outputJ = netJ(xz_input).view(b_size, -1)
          Sx = scoreF(outputF)
          Sz = scoreH(outputH)
          Sxz = scoreJ(outputJ)
          #--Loss_EG = y(Sx + Sxz + Sz) where y in {-1, 1}--
          Loss_EG_fake = F.sigmoid(Sx + Sxz + Sz + .000001).view(-1) #trick
          #Loss_EG_fake = F.sigmoid(-1 * (Sx + Sxz + Sz + .000001)).view(-1) #actual
          errG = criterion(Loss_EG_fake, label)
          errG.backward(retain_graph=True) # (may be wrong, but moving forward) 'retain_graph=True' to clear "RuntimeError: Trying to backward through the graph a second time, but the buffers have already been freed."
          # Update G
          optimizerG.step()
          
          #--updating encoder--
          netE.zero_grad()
          label.fill_(fake_label) #trick
          #label.fill_(real_label) #actual
          outputF = netD(real_cpu).view(b_size, -1)
          outputH = netH(E_output).view(b_size, -1)
          xz_input = torch.cat((outputF, outputH), -1)
          outputJ = netJ(xz_input).view(b_size, -1)
          Sx = scoreF(outputF)
          Sz = scoreH(outputH)
          Sxz = scoreJ(outputJ)
          #--Loss_EG = y(Sx + Sxz + Sz) where y in {-1, 1}--
          Loss_EG_real = F.sigmoid(-1 * (Sx + Sxz + Sz + .000001)).view(-1) #trick
          #Loss_EG_real = F.sigmoid(Sx + Sxz + Sz + .000001).view(-1) #actual
          errE = criterion(Loss_EG_real, label)
          errE.backward(retain_graph=True)
          # Update E
          optimizerE.step()
          
          if i % 50 == 0:
              print('[%d/%d][%d/%d]\tLoss_XZ: %.4f\tLoss_E: %.4f\tLoss_G: %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                       errXZ.item(), errE.item(), errG.item()))

          # Save Losses for plotting later
          E_losses.append(errE.item()) #added
          G_losses.append(errG.item())
          D_losses.append(errXZ.item()) #now is XZ

          # Check how the generator is doing by saving G's output on fixed_noise
          if (iters % 50 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
              with torch.no_grad():
                  fake = netG(fixed_noise).detach().cpu()
              img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
              if print_100:
                # Plot the fake images from this epoch
                plt.subplot(1,2,2)
                plt.axis("off")
                plt.title("Fake Images")
                plt.imshow(np.transpose(img_list[-1],(1,2,0)))
                plt.show()
              
          # Check how the encoder is doing by saving E's output on "last real" #added (untested), probably need to hold back a 'constant_real' to feed in
          #--!!--NOTE: is now netG(netE(real_cpu)), not anything interesting yet in current testing
          if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
              with torch.no_grad():
                  E_x = netG(netE(real_cpu)).detach().cpu()
              E_img_list.append(vutils.make_grid(E_x, padding=2, normalize=True))
              
          # Save progress (untested)
          if (iters % 10 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            checkpoint(iters, netG, netD, netE, netH, netJ, scoreF, scoreH, scoreJ)
            print("checkpoint created.(check google drive)")
            
        iters += 1


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
if BigBiGAN:
  plt.plot(E_losses,label="E")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())


# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()

if BigBiGAN:
  # Plot the encoder images from the last epoch
  # --!!!--Notice: this is now G(E(x))
  plt.subplot(1,2,2)
  plt.axis("off")
  plt.title("Encoder Images")
  plt.imshow(np.transpose(E_img_list[-1],(1,2,0))) #empty atm, need to keep and collect encoder outs
  plt.show()


# Code to create an animation from your sampled interpolation

#@title Interpolation solution {display-mode: "form"}

imgs = []
values = np.arange(0, 1, 1./64)
for idx in range(nz):
  z = np.random.uniform(-0.2, 0.2, size=(nz))
  z_sample = np.tile(z, (64, 1))
  for kdx, z in enumerate(z_sample):
    z[idx] = values[kdx]
  #print("z: ", z_sample.shape)
  with torch.no_grad():
    if DCGAN:
      fake = netG(torch.from_numpy(z_sample.reshape((64, 100, 1, 1))).float().to(device)).detach().cpu()
    else: #BigBiGAN
      fake = netG(torch.from_numpy(z_sample.reshape((64, 196, 1, 1))).float().to(device)).detach().cpu()
  imgs.append(vutils.make_grid(fake, padding=2, normalize=True))

#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in imgs[:25]]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())

#end
