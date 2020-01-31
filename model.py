import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pickle
import config
import nltk
from functools import partial 
nltk.download('stopwords')
nltk.download('punkt')

# ===== LOAD DATA =====
with open(config.dictionary_path, 'rb') as f:
  my_dictionary = pickle.load(f)

with open(config.dataset_path, 'rb') as f:
  dataset_flickr30k = pickle.load(f)

# ===== LOAD KERNEL INIT =====
from gensim.models import KeyedVectors
model_word = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

NUMB_WORDS = len(my_dictionary)
NUMB_FT = 300
kernel_init = np.zeros((NUMB_WORDS, NUMB_FT))
for idx, word in enumerate(my_dictionary):
  try:
    word_2_vec_ft = model_word.word_vec(word)
    word_2_vec_ft = np.reshape(word_2_vec_ft, (1, NUMB_FT))
  except KeyError:
    word_2_vec_ft = np.random.rand(1, NUMB_FT)
  kernel_init[idx,:] = word_2_vec_ft
  
# ===== RESIDUAL BLOCK =====
class Conv2dAuto(nn.Conv2d):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if self.kernel_size[1] % 2 == 0:
      self.dilation = (1, 2)
    self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)
conv1x1 = partial(Conv2dAuto, kernel_size=1, bias=False)     

def activation_func(activation):
  return  nn.ModuleDict([
    ['relu', nn.ReLU(inplace=True)],
    ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
    ['selu', nn.SELU(inplace=True)],
    ['none', nn.Identity()]
  ])[activation]

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, activation='relu'):
    super().__init__()
    self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
    self.blocks = nn.Identity()
    self.activate = activation_func(activation)
    self.shortcut = nn.Identity()   

  def forward(self, x):
    residual = x
    if self.should_apply_shortcut: residual = self.shortcut(x)
    x = self.blocks(x)
    x += residual
    x = self.activate(x)
    return x

  @property
  def should_apply_shortcut(self):
    return self.in_channels != self.out_channels

class ResNetResidualBlock(ResidualBlock):
  def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv1x1, *args, **kwargs):
    super().__init__(in_channels, out_channels, *args, **kwargs)
    self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
    self.shortcut = nn.Sequential(
        nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                  stride=self.downsampling, bias=False),
        nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None
      
      
  @property
  def expanded_channels(self):
    return self.out_channels * self.expansion
  
  @property
  def should_apply_shortcut(self):
    return self.in_channels != self.expanded_channels

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
  return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))

class ResNetBasicBlock(ResNetResidualBlock):
  """
  Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
  """
  expansion = 1
  def __init__(self, in_channels, out_channels, *args, **kwargs):
    super().__init__(in_channels, out_channels, *args, **kwargs)
    self.blocks = nn.Sequential(
        conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
        activation_func(self.activation),
        conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
    )

# For Resnet 50 upper
class ResNetBottleNeckBlock(ResNetResidualBlock):
  expansion = 4
  def __init__(self, in_channels, out_channels, *args, **kwargs):
    super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
    self.blocks = nn.Sequential(
        conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1, stride=self.downsampling),
          activation_func(self.activation),
          conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=(1, 2)),
          activation_func(self.activation),
          conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
    )

class ResNetLayer(nn.Module):
  """
  A ResNet layer composed by `n` blocks stacked one after the other
  """
  def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
      super().__init__()
      # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
      downsampling = (1, 2) if in_channels != out_channels else (1, 1)
      self.blocks = nn.Sequential(
          block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
          *[block(out_channels * block.expansion, 
                  out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
      )

  def forward(self, x):
      x = self.blocks(x)
      return x

class Image_Branch(nn.Module):
  def __init__(self, backbone_trainable = False):
    super(Image_Branch, self).__init__()
    resnet = models.resnet50(pretrained=True)
    modules = list(resnet.children())[:-2]
    self.resnet = nn.Sequential(*modules)
    for p in self.resnet.parameters():
      p.requires_grad = backbone_trainable

    self.a1 = nn.AdaptiveAvgPool2d((1, 1))
  
    self.after = nn.Sequential(
      nn.Linear(2048, 2048),
      nn.BatchNorm1d(2048),
      nn.ReLU(),
      nn.Dropout(p=0.5),
      nn.Linear(2048, 2048)
    )

  def forward(self, x):
    x = self.resnet(x)
    x = self.a1(x)
    x = x.view(x.size(0), -1)
    x = self.after(x)
    return x

class Text_Branch(nn.Module):
  def __init__(self, in_channels=1, blocks_sizes=[64, 128, 256, 512], deepths=[3, 4, 6, 3],
               activation='relu', block=ResNetBottleNeckBlock, *args, **kwargs):
    super(Text_Branch, self).__init__()
    self.blocks_sizes = blocks_sizes
    self.gate = nn.Sequential(
      nn.Conv2d(in_channels=in_channels, out_channels=300, kernel_size=1, stride=1, bias=False),
      nn.BatchNorm2d(300),
      activation_func('relu')
    )

    self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
    self.blocks = nn.ModuleList([ 
      ResNetLayer(300, blocks_sizes[0], n=deepths[0], activation=activation, 
                  block=block,*args, **kwargs),
      *[ResNetLayer(in_channels * block.expansion, 
                    out_channels, n=n, activation=activation, 
                    block=block, *args, **kwargs) 
        for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
    ])
    
    self.a1 = nn.AdaptiveAvgPool2d((1, 1))

    self.after = nn.Sequential(
      nn.Linear(2048, 2048),
      nn.BatchNorm1d(2048),
      nn.ReLU(),
      nn.Dropout(p=0.5),
      nn.Linear(2048, 2048)
    )

  def forward(self, x):
    x = self.gate(x)
    for block in self.blocks:
      x = block(x)
    x = self.a1(x)
    x = x.view(x.size(0), -1)
    x = self.after(x)
    return x

class DualPath(nn.Module):
  def __init__(self, nimages, nwords=len(my_dictionary), backbone_trainable = False, initial_word2vec=True):
    super(DualPath, self).__init__()
    self.image_branch = Image_Branch(backbone_trainable=backbone_trainable)
    self.text_branch = Text_Branch(in_channels=nwords)
    self.sharew = nn.Linear(2048, nimages, bias=False)
    if initial_word2vec:
      with torch.no_grad():
        K = torch.tensor(kernel_init.T)
        K = nn.Parameter(torch.unsqueeze(torch.unsqueeze(K, 2),3))
        self.text_branch.gate[0].weight.data = K
  
  def forward(self, x_img, x_txt):
    f_img = self.image_branch(x_img)
    f_txt = self.text_branch(x_txt)
    y_img = self.sharew(f_img)
    y_txt = self.sharew(f_txt)
    return y_img, y_txt, f_img, f_txt

# ===== LOSS =====
criterion = nn.CrossEntropyLoss()
cosine_simi = nn.CosineSimilarity()
def loss_func(pred_img, pred_txt, lbl, f_img, f_txt, alpha=1, lamb_0=1, lamb_1=1, lamb_2=1):
  batch_size = len(lbl)
  L_img = criterion(pred_img[0:batch_size], lbl)
  L_txt = criterion(pred_txt[0:batch_size], lbl)
  instance_loss = lamb_0*L_img + lamb_1*L_txt

  image_cosine = alpha - (cosine_simi(f_img[:batch_size], f_txt[:batch_size]) - cosine_simi(f_img[:batch_size], f_txt[batch_size:]))
  text_cosine = alpha - (cosine_simi(f_txt[:batch_size], f_img[:batch_size]) - cosine_simi(f_txt[:batch_size], f_img[batch_size:]))
  image_rank_loss = torch.max(torch.tensor(0.), image_cosine)
  text_rank_loss = torch.max(torch.tensor(0.), text_cosine)
  ranking_loss = lamb_2*(torch.sum(image_rank_loss + text_rank_loss))/batch_size

  # loss = instance_loss + ranking_loss

  return L_img, L_txt, ranking_loss

# ===== CREATE MODEL =====
def load_model_testing(nimages, nwords=len(my_dictionary), pretrained_path=''):
  model = DualPath(nimages, nwords, False, False)
  model.load_state_dict(torch.load(pretrained_path))
  model.eval() # Use this inly for test --> remove dropout and batchnorm
  return model

def load_model_training(nimages, nwords=len(my_dictionary), backbone_trainable=False, pretrained_path=''):
  if pretrained_path != '':
    model = DualPath(nimages, nwords, backbone_trainable, False)
    optimizer = optim.Adam(model.parameters())
    checkpoint = torch.load(pretrained_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']