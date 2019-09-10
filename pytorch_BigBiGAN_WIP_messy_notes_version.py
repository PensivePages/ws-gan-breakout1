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
  nz = 196#<-1860/2#100

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


#[Q]^ should F, H, J all reduce to 1 for BCE? Or, should only J recuder to 1? Or, should none of them reduce to 1?
#^ I'm now wondering if "x + xz + z" is symbolic of seperate training of: x on real, z on fake, and xz on both...? They say they train together, so...no? But since the main point is to train the generator wouldn't this do that?

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
        return F.tanh(self.output(out))#.view(batch_size, nz, 1, 1) #put tanh here instead of sigmoid


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


#-the following change in 'criterion' will come from and inferance drawn from a few following sources
# the inferance is that: BigBiGAN mean's the hinge loss function for each of these in this sum (x + xz + z)
# such that F_out and H_out feed into J without the mean(hinge) to get J_out
# then they do one of 2 things: (Note: the paper explicitely ruled out, 'hinge(x + xz + z)' )
# |mean(hinge(F_out)) + mean(hinge(J_out)) + mean(hinge(H_out))| 
# or 
# |mean(hinge(F_out) + hinge(J_out) + hinge(H_out))|
# here are the things that lead me to infer that this is how BigBiGAN handles the loss:
# BigGAN code comment "The output dimension is typically 1, but may be larger if we're e.g. turning this into a VAE with an inference output" from: https://github.com/ajbrock/BigGAN-PyTorch/blob/master/BigGANdeep.py
# BigGAN loss code ex: "torch.mean(F.relu(1. - dis_real)) and torch.mean(F.relu(1. + dis_fake))" from: https://github.com/ajbrock/BigGAN-PyTorch/blob/master/losses.py
# VAE article quotes are: (from: https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)
# "...reconstruction log-likelihood..." (which is referenced in the BigBiGAN paper) 
# ^they also call it "or expected negative log-likelihood of the ii-th datapoint" 
# ^and it shows the notation from the paper, where I now believe th 'E' stands for "expected" but may also stand in for "1/N*sum from i to N" ie: the "average", https://en.wikipedia.org/wiki/Maximum_likelihood_estimation (crtl+f "expected")
# "The loss function of the variational autoencoder..."
# "...we can decompose the loss function into only terms that depend on a single datapoint l_i. The total loss is then sum (from i to N) l_i for N total datapoints"
# ^I believe is saying the "reconstruction log-likelihood" values at each datapoint are summed
# right before the above quote is this caveat, that I don't follow and may refer to NNets, "Because there are no global representations that are shared by all datapoints"
# NOTE: for now I'm going to have F, H and J outputs same shape matricies.
# ^However, I could have J output to size 1 and only have to sum E_out and H_out (which might be better for leanring? may need to experiment)

# Hinge Loss (from: https://github.com/ajbrock/BigGAN-PyTorch/blob/master/losses.py)
def loss_hinge_dis(dis_real, dis_fake): # reversed input cuz THATS WHY! :-P
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake
# def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
  # loss = torch.mean(F.relu(1. - dis_real))
  # loss += torch.mean(F.relu(1. + dis_fake))
  # return loss
  
def loss_hinge_gen(dis_fake):
  loss = -torch.mean(dis_fake)
  return loss

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
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999)) # F
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
    
    
load = True
if load: # NOTICE: (untested)
  load_checkpoint(netG, 'G.pkl')
  load_checkpoint(netE, 'E.pkl')
  load_checkpoint(netD, 'D.pkl') # F
  load_checkpoint(netH, 'H.pkl')
  load_checkpoint(netJ, 'J.pkl')
  load_checkpoint(scoreF, 'Sx.pkl')
  load_checkpoint(scoreJ, 'Sxz.pkl')
  load_checkpoint(scoreH, 'Sz.pkl')
  print("saved values loaded.")


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
                #print_i += 1
        
        elif BigBiGAN: #used elif for legability and ctrl + f searchability
          #print("BigBiGAN is Under construction")
          #break
          ############################
          # (1) Update D network: maximize log(J(F(x), H(E(x)))) + log(1 - J(F(G(z)), H(z)))
          ###########################
          
          #---"Autoencoder" Section | enc, gen
          
          #--run the encoder--
          ## Training components x -> E(x) batch
          #netE.zero_grad() #removing to stay true to the pattern >.> especially since I don't really know pytorch >.> lol
          # Format batch
          #print("data[0].size(): ", data[0].size())
          real_cpu = data[0].to(device)
          b_size = real_cpu.size(0)
#           if b_size != batch_size:
#             continue
          #label = torch.full((b_size,), real_label, device=device) #commenting out, will use downstream
          # Forward pass real batch through D
          E_output = netE(real_cpu).view(b_size, -1)
          # Calculate loss on all-real batch
          #errE_real = criterion(output, label) #commenting out, need to move this and add inputs
          # Calculate gradients for D in backward pass
          #errD_real.backward() #commenting out, will back prop after added inputs
          #D_x = output.mean().item() #commenting out, if I still track this it will be 'D_(x, G(z))'

          #--run the generator--
          ## Training components z -> G(z) batch
          # Generate batch of latent vectors
          #netG.zero_grad() #added #removing to stay true to the pattern >.> especially since I don't really know pytorch >.> lol
          noise = torch.randn(b_size, nz, 1, 1, device=device)
          # Generate fake image batch with G
          G_output = fake = netG(noise)
          #label.fill_(fake_label) #commenting out, will use downstream
          # Classify all fake batch with D
          #output = netD(fake.detach()).view(-1) #commented out, will 'D' downstream
          # Calculate D's loss on the all-fake batch
          #errD_fake = criterion(output, label) #commenting out, need to move this and add inputs
          # Calculate the gradients for this batch
          #errD_fake.backward() #commenting out, will back prop after added inputs
          #D_G_z1 = output.mean().item() #commenting out, if I still track this it will be 'D_(E(x), z)'
          # Add the gradients from the all-real and all-fake batches
          #errD = errD_real + errD_fake #commenting out, if I still track this it will be 'err_(x, G(z))' + 'err_(E(x), z)' + 'err_((x, G(z)), (E(x), z))'
          # Update D
          #optimizerD.step() #commenting out, optimize downstream
          
          #--NOTICE--this whole section seems to be a misinterpretation of the paper, no cancatenation is done, until J?
#           #--combine unary componets X = (x, G(z)), Z = (E(x), z)
#           #-Notice the patterns (encoder, generator) for unary compents of BigBiGAN
#           #print("real_cpu.size(): ", real_cpu.size())
#           #print("G_output.size(): ", G_output.size())
#           x_data = torch.cat((real_cpu, G_output), -1) #assuming NCWH, so concat'ing on H
#           #print("x_data .size(): ", x_data .size())
#           noise = noise.view(b_size, -1) #added .view(-1), taking latent space, E_output and now noise, and then flattening it for its discriminator (H)
#           #print("E_output.size(): ", E_output.size())
#           #print("noise.size(): ", noise.size())
#           z_latents = torch.cat((E_output, noise), -1) #assuming NCWH, so concat'ing on H
#           #print("z_latents.size(): ", z_latents.size())
#           #xz_comp = concat(x_comp, z_comp) #mistaken concept on my part I think
#           #--combine joint (xz) componets (x, z) = (x, E(x)), (x, z) = (G(z), z)
#           #-Notice the patterns (enc, enc) and (gen, gen) for BiGAN, which end up being (enc, gen) for BigBiGAN
#           #real_comp = torch.cat((real_cpu, E_output), -1) #assuming NCWH, so concat'ing on H
#           #fake_comp = torch.cat((G_output, noise), -1) #assuming NCWH, so concat'ing on H
          
          #[QUESTION] do we use matmul to get xz to shape x (or z, so can add: x + xz + z)? Or, do we splice add x + xz[:end-of-x], z + xz[start-of-z:]
          
          #---Unary Discriminators Section | F (which is 'D' here), H
          
          #--!!--NOTICE: currently training only joint (so no unaries, ie: BiGAN)--!!--
          ## Train with all-real batch
          #--run F on real (ie: X = (x, G(z)))--
          netD.zero_grad()
          # Format batch
          #xz_real_cpu = xz_comp.to(device) #mistaken concept on my part I think
          x_real_cpu = E_output.to(device)
          b_size = x_real_cpu.size(0)
          label = torch.full((b_size,), real_label, device=device)
          # Forward pass real batch through D
          #print("x_real_cpu.size(): ", x_real_cpu.size())
          F_output_real = netD(real_cpu).view(b_size, -1)
          Sx_real = scoreF(F_output_real)
          F_output_fake = netD(G_output).view(b_size, -1)
          Sx_fake = scoreF(F_output_fake)
          #print("x_output.size(): ", x_output.size())
          # Calculate loss on all-real batch
          #errD_x_real = criterion(x_output, label) #commenting out, loss downstream
          # Calculate gradients for D in backward pass
          #errD_x_real.backward() #commenting out, backprop downstream
          #D_E_z1 = F_output_real.mean().item() #placeholder, wrong cuz missing fake
          #D_G_z1 = F_output_fake.mean().item() #wrong, not messing with

          ## Train with all-fake batch
          #--run H on fake (ie: Z = (E(x), z))--
          netH.zero_grad() #added
          # Generate batch of latent vectors
          #noise = torch.randn(b_size, nz, 1, 1, device=device)
          # Generate fake image batch with G
          #xz_fake = netD(xz_comp) #mistaken concept on my part I think
          #xz_fake = netD(z_latents) #improper use of artifact of DCGAN, latents already known, netD should have been netG
          label.fill_(fake_label)
          # Classify all fake batch with D
          #print("z_latents.size(): ", z_latents.size())
          H_output_real = netH(E_output.detach()).view(b_size, -1)
          Sz_real = scoreH(H_output_real)
          noise = noise.view(b_size, -1) #reshape for H
          H_output_fake = netH(noise.detach()).view(b_size, -1)
          Sz_fake = scoreH(H_output_fake)
          #print("z_output.size(): ", z_output.size())
          # Calculate D's loss on the all-fake batch
          #errD_z_fake = criterion(z_output, label) #commenting out, loss downstream
          # Calculate the gradients for this batch
          #errD_z_fake.backward() #commenting out, backprop downstream
          #H_output = H_output_real + H_output_fake
          #D_H_z1 = H_output_real.mean().item() #placeholder, wrong cuz missing fake
          #D_H_z1 = H_output_fake.mean().item() #wrong, not messing with
          # Add the gradients from the all-real and all-fake batches
          #errD = errD_xz_real + errD_xz_fake  # this section is wrong, I'll fix it later
          # Update D
          #optimizerD.step() #commenting out, thi section is no longer seing same discriminators, or data, moved downstream and will be 'J'
          
          #---Joint (xz) Discriminator | J
#           print("F_output_real.size(): ", F_output_real.size())
#           print("H_output_real.size(): ", H_output_real.size())
#           print("F_output_fake.size(): ", F_output_fake.size())
#           print("H_output_fake.size(): ", H_output_fake.size())
#           raise NotImplementedError
          xz_input_real = torch.cat((F_output_real, H_output_real), -1)
          xz_input_fake = torch.cat((F_output_fake, H_output_fake), -1)
          
          #--run J on real (ie: X = F_outputs)--
          netJ.zero_grad()
          # Format batch
          #xz_real_cpu = xz_comp.to(device) #mistaken concept on my part I think
          xz_real_cpu = xz_input_real.to(device)
          b_size = xz_real_cpu.size(0)
          #label = torch.full((b_size,), real_label, device=device)
          # Forward pass real batch through D
          #print("xz_real_cpu.size(): ", xz_real_cpu.size())
          J_output_real = netJ(xz_real_cpu)#.view(b_size, -1)
          Sxz_real = scoreJ(J_output_real)
          # Calculate loss on all-real batch
          #errD_xz_real = criterion(xz_output_real, label) #consolidating real anf ake, using hinge loss
          # Calculate gradients for D in backward pass
          #errD_xz_real.backward(retain_graph=True) #moving to after combind loss
          #D_X_z0 = xz_output_real.mean().item() #moving to after combind loss
          
          #--run J on fake (ie: Z = H_outputs)--
          # Generate batch of latent vectors
          #noise = torch.randn(b_size, nz, 1, 1, device=device)
          # Generate fake image batch with G
          #xz_fake = netD(xz_comp) #mistaken concept on my part I think
          #xz_fake = netD(z_latents) #improper use of artifact of DCGAN, latents already known, netD should have been netG
          #label.fill_(fake_label)
          # Classify all fake batch with D
          #print("z_output.size(): ", z_output.size())
          J_output_fake = netJ(xz_input_fake.detach())#.view(b_size, -1)
          Sxz_fake = scoreJ(J_output_fake)
          #J_output = J_output_real + J_output_fake #wrong, not messing with
          #D_J = J_output.mean().item() #wrong, not messing with
          
#           #--Loss_EG = y(Sx + Sxz + Sz) where y in {-1, 1}-- (moving downstream to E and G updates)
#           Loss_EG_real = Sx + Sxz_real + Sz
#           Loss_EG_fake = -1 * (Sx + Sxz_fake + Sz)
#           Loss_EG_real.backward(retain_graph=True)
#           Loss_EG_fake.backward(retain_graph=True)
          
          #--Loss_D  = hinge(y(Sx)) + hinge(y(Sxz)) + hinge(y(Sz)) where y in {-1, 1}
          #Loss_D_real = F.relu(2. - Sx_real) + F.relu(2. - Sxz_real) + F.relu(2. - Sz_real) + .000001#).view(-1)
          #Loss_D_fake = F.relu(2. + Sx_fake) + F.relu(2. + Sxz_fake) + F.relu(2. + Sz_fake) + .000001#).view(-1)
          Loss_D_real = F.sigmoid(F.relu(2. - Sx_real) + F.relu(2. - Sxz_real) + F.relu(2. - Sz_real) + .000001).view(-1)
          Loss_D_fake = F.sigmoid(F.relu(2. + Sx_fake) + F.relu(2. + Sxz_fake) + F.relu(2. + Sz_fake) + .000001).view(-1)
          #print("Loss_D_real.size(): ", Loss_D_real.size())
          #print("Loss_D_fake.size(): ", Loss_D_fake.size())
          # Calculate the gradients for this batch
          label = torch.full((b_size,), real_label, device=device)
          Loss_D_real = criterion(Loss_D_real, label)
          Loss_D_real.backward(retain_graph=True) #added
          #D_X_z0 = J_output_real.mean().item() #think about if 'J_output_real' wrong later (Sxz_real?)
          label.fill_(fake_label)
          Loss_D_fake = criterion(Loss_D_fake, label)
          Loss_D_fake.backward(retain_graph=True)
          #D_Z_z0 = J_output_fake.mean().item() #think about if 'J_output_fake' wrong later (Sxz_fake?)
          # Add the gradients from the all-real and all-fake batches
          errXZ = Loss_D_real + Loss_D_fake
          optimizerSxz.step()
          optimizerSz.step()
          optimizerSx.step()
          optimizerJ.step()
          optimizerH.step()
          optimizerD.step() #D is F
          
#           #--initial attempts version-- (commenting out, switch to Loss_EG and Loss_D paradigm)
#           # Calculate D's loss on the all-fake batch
#           #errD_xz_fake = criterion(xz_output_fake, label) #consolidating real anf ake, using hinge loss
#           errJ_xz_real_real, errJ_xz_fake = loss_hinge_dis(F_output + J_output_real, H_output) # run real though F and J with only fake from H
#           errJ_xz_real, errJ_xz_fake_fake = loss_hinge_dis(F_output, J_output_fake + H_output) # run on only real from F with fake on J with H
#           errJ_xz_real = errJ_xz_real + errJ_xz_real_real # set real = F + (F + J_real)
#           errJ_xz_fake = errJ_xz_fake + errJ_xz_fake_fake # set fake = H + (H + J_fake)
#           # Calculate the gradients for this batch
#           errJ_xz_real.backward(retain_graph=True) #added
#           D_X_z0 = xz_output_real.mean().item() #added
#           errJ_xz_fake.backward(retain_graph=True)
#           D_Z_z0 = xz_output_fake.mean().item()
#           # Add the gradients from the all-real and all-fake batches
#           #errD = errD_xz_real + errD_xz_fake # this section is wrong, I'll fix it later
#           errJ = errJ_xz_real + errJ_xz_fake
#           # Update J, H, F [Q] does pytorch update all when I update J, or do I need to do H and F? guess we'll see :-P
#           optimizerJ.step()
#           optimizerH.step()
#           optimizerD.step() #D is F
          
#           #--version of E training (curiosity reasons only, not using atm) #added
#           label = torch.full((b_size,), real_label, device=device)
#           errE_real = criterion(E_output , label)
#           # Calculate gradients for D in backward pass
#           errE_real.backward()
#           D_xGz = output.mean().item() #commenting out, if I still track this it will be 'D_(x, G(z))'
#           optimizerE.step() #added
          
#           #--version of G training (curiosity reasons only, not using atm) #added
#           label.fill_(fake_label)
#           # Classify all fake batch with D
#           output = netD(fake.detach()).view(-1)
#           # Calculate D's loss on the all-fake batch
#           errG_fake = criterion(G_output, label) #commenting out, need to move this and add inputs
#           # Calculate the gradients for this batch
#           errG_fake.backward() #commenting out, will back prop after added inputs
#           #D_Exz = output.mean().item() #commenting out, if I still track this it will be 'D_(E(x), z)'
#           # Add the gradients from the all-real and all-fake batches
#           #errD = errD_real + errD_fake #commenting out, if I still track this it will be 'err_(x, G(z))' + 'err_(E(x), z)' + 'err_((x, G(z)), (E(x), z))'
#           optimizerG.step() #added

          ############################
          # (2) Update G (and E) network: maximize log(D(G(z))) (and log(D(E(x)))?)
          ###########################
          #--updating generator--
          netG.zero_grad()
          label.fill_(real_label)  #trick # fake labels are real for generator cost
          #label.fill_(fake_label) #actual
          # Since we just updated D, perform another forward pass of all-fake batch through D
          outputF = netD(G_output).view(b_size, -1) #added #commented out, training only on fake side
          outputH = netH(noise).view(b_size, -1) #replaced 'fake' with 'G_output', notation change mainly, but remeber that the generator is actually trying to produce 'real'
          xz_input = torch.cat((outputF, outputH), -1)
          outputJ = netJ(xz_input).view(b_size, -1) #add in J
          Sx = scoreF(outputF)
          Sz = scoreH(outputH)
          Sxz = scoreJ(outputJ)
          # Calculate G's loss based on this output
          #errG = criterion(output, label) #changed to hinge loss below
          #errG = loss_hinge_gen(output1 + output2) #change to output 1 and 2, was only running off J
          #--Loss_EG = y(Sx + Sxz + Sz) where y in {-1, 1}--
          Loss_EG_fake = F.sigmoid(Sx + Sxz + Sz + .000001).view(-1) #trick
          #Loss_EG_fake = F.sigmoid(-1 * (Sx + Sxz + Sz + .000001)).view(-1) #actual
          errG = criterion(Loss_EG_fake, label)
          # ^may want to keep J out and just run on H
          # Calculate gradients for G
          errG.backward(retain_graph=True) # (may be wrong, but moving forward) 'retain_graph=True' to clear "RuntimeError: Trying to backward through the graph a second time, but the buffers have already been freed."
          #D_G_z2 = outputF.mean().item()
          # Update G
          optimizerG.step()
          
          #--updating encoder--
          netE.zero_grad()
          label.fill_(fake_label) #trick
          #label.fill_(real_label) #actual
          # Since we just updated D, perform another forward pass of all-fake batch through D
          outputF = netD(real_cpu).view(b_size, -1) #replaced 'fake' with 'E_output', notation change mainly, but remeber that the encoder is actually trying to produce 'fake' (or, more accurately, "latent space representations")
          outputH = netH(E_output).view(b_size, -1) #added #commented out, training only on real side
          xz_input = torch.cat((outputF, outputH), -1)
          outputJ = netJ(xz_input).view(b_size, -1) #add in J
          Sx = scoreF(outputF)
          Sz = scoreH(outputH)
          Sxz = scoreJ(outputJ)
          # Calculate G's loss based on this output
          #errE = criterion(output, label) #changed to hinge loss below
          #errE = loss_hinge_enc(output1 + output2) #change to output 1 and 2, was only running off J
          #--Loss_EG = y(Sx + Sxz + Sz) where y in {-1, 1}--
          Loss_EG_real = F.sigmoid(-1 * (Sx + Sxz + Sz + .000001)).view(-1) #trick
          #Loss_EG_real = F.sigmoid(Sx + Sxz + Sz + .000001).view(-1) #actual
          errE = criterion(Loss_EG_real, label)
          # ^may want to keep J out and just run on F
          # Calculate gradients for G
          errE.backward(retain_graph=True)
          #D_E_z2 = outputF.mean().item()
          # Update E
          optimizerE.step()

          # Output training stats
#           if i % 50 == 0:
#               print('[%d/%d][%d/%d]\tLoss_XZ: %.4f\tLoss_E: %.4f\tLoss_G: %.4f\tJ(F(x), H(E(x))): %.4f / %.4f / %.4f\tJ(F(G(z)), H(z)): %.4f / %.4f / %.4f'
#                     % (epoch, num_epochs, i, len(dataloader),
#                        errXZ.item(), errE.item(), errG.item(), D_X_z0, D_X_z1, D_E_z2, D_Z_z0, D_Z_z1, D_G_z2)) # decide to just track losses for the moment#
          if i % 50 == 0:
              print('[%d/%d][%d/%d]\tLoss_XZ: %.4f\tLoss_E: %.4f\tLoss_G: %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                       errXZ.item(), errE.item(), errG.item()))

          # Save Losses for plotting later
          E_losses.append(errE.item()) #added
          G_losses.append(errG.item())
          D_losses.append(errXZ.item())
          #H_losses.append(errH.item()) #not yet in use
          #J_losses.append(errD.item()) #not yet in use | 'F'

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
                #print_i += 1
              
          # Check how the encoder is doing by saving E's output on "last real" #added (untested), probably need to hold back a 'constant_real' to feed in
          #--!!--NOTE: is now netG(netE(real_cpu))
          if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
              with torch.no_grad():
                  E_x = netG(netE(real_cpu)).detach().cpu()
              E_img_list.append(vutils.make_grid(E_x, padding=2, normalize=True))
              
          # Save progress (untested)
          if (iters % 10 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            checkpoint(iters, netG, netD, netE, netH, netJ, scoreF, scoreJ, scoreH)
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
