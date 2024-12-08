import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

# TODO Task 1c - Implement a SimpleBNConv-----------------------------
## Fully connected layers --------------------------------------------

class MLP(nn.Module):
  def __init__(self, input_size, output_size=7):
      super(MLP,self).__init__()
      
      self.FCL = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size, 32),
        nn.ReLU(),
        nn.Linear(32, output_size)
      )
    
  def forward(self, cv_layers):
    return self.FCL(cv_layers)

## Convolutional layers -----------------------------------------------
class SimpleBNConv(nn.Module):
  def __init__(self, device):
    super(SimpleBNConv, self).__init__()
    self.feature_extractor = nn.Sequential(      
      ## Convolutional layers
      nn.Conv2d(3, 8, 3, padding = 1),
      nn.ReLU(),
      nn.BatchNorm2d(8),
      nn.MaxPool2d(2),
      
      nn.Conv2d(8, 16, 3, padding = 1),
      nn.ReLU(),
      nn.BatchNorm2d(16),
      nn.MaxPool2d(2),

      nn.Conv2d(16, 32, 3, padding = 1),
      nn.ReLU(),
      nn.BatchNorm2d(32),
      nn.MaxPool2d(2),

      nn.Conv2d(32, 64, 3, padding = 1),
      nn.ReLU(),
      nn.BatchNorm2d(64),
      nn.MaxPool2d(2),

      nn.Conv2d(64, 128, 3, padding = 1),
      nn.ReLU(),
      nn.BatchNorm2d(128),
      nn.MaxPool2d(2)
    )
    self.mlp = MLP(input_size = 128*14*18, output_size=7)
    ## input_size = 128*9*9 for 299*299

  def forward(self, input_tensor):
    cv_features = self.feature_extractor(input_tensor)
    #print("features shape:", cv_features.shape)
    return self.mlp(cv_features)



# TODO Task 1f - Create a model from a pre-trained model from the torchvision
#  model zoo. -----------------------------------------------------------

class ResNet18(nn.Module):
  def __init__(self, device):
    super(ResNet18, self).__init__()

    # Downloading pre-trained resnet18
    resnet18 = models.resnet18(pretrained=True)

    # Freezing model weights
    for parameter in resnet18.parameters():
      parameter.requires_grad = True

    # Finding variable name of the last linear layer = fc with 512 inputs
    #print(resnet18)
    
    # Unfreezing the weights for the last layer
    #for parameter in resnet18.fc.parameters():
    #  parameter.requires_grad = True

    # Changing the final linear layer
    resnet18.fc = nn.Linear(512, 7)

    self.resnet = resnet18.to(device)
    #print(self.resnet)

  def forward(self, input_tensor):
    return self.resnet(input_tensor)
  
    ## Note all pre trained models expect input images to be normalised
    # in the same way. Therefore please see normalisation in datasets.py

# TODO Task 1f - Create your own models

# Five cropping for the challenge 1e ------------------------------------
class FiveCropModel(nn.Module):
  def __init__(self,device):
    super(FiveCropModel, self).__init__()

    #Use the same model archictecture as SimpleBNConv
    self.device = device
    self.five_crop_conv_layers = nn.Sequential(      
      ## Convolutional layers (starts with 15 as 3channels*5crops)
      nn.Conv2d(15, 32, 3, padding = 1),
      nn.ReLU(),
      nn.BatchNorm2d(32),
      nn.MaxPool2d(2),
      
      nn.Conv2d(32, 64, 3, padding = 1),
      nn.ReLU(),
      nn.BatchNorm2d(64),
      nn.MaxPool2d(2),

      nn.Conv2d(64, 128, 3, padding = 1),
      nn.ReLU(),
      nn.BatchNorm2d(128),
      nn.MaxPool2d(2)
    )
    self.five_crop_fc_layers = MLP(input_size = 128*25*37, output_size=7)
    
  def forward(self, five_crop_tensors):
    #Check for shape
    #print(five_crop_tensors.shape)
    
    # Passing through the convolutional layers
    features = self.five_crop_conv_layers(five_crop_tensors)
    #print(features.shape)
    
    # Passing through the fully connected layers and returning
    return self.five_crop_fc_layers(features)

    # -------------------------------------------------------------------