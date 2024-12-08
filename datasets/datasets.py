
import collections
import csv
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

## LesionDataset implementation ------------------------------------------
class LesionDataset:
  # initialises the image directory, the csv and img names into a list
  def __init__(self, img_dir, labels_fname, augment = False):
    
    self.img_dir = Path(img_dir)
    self.images = pd.read_csv(labels_fname) 
    self.image_paths = []
    self.labels = self.images['Label'].tolist()
    self.augment = augment

    for image in self.images['image']:
      image_path = self.img_dir / (image + '.jpg')
      self.image_paths.append(image_path)
    
    #print(self.labels[0:5])
    #print(self.images.head())
    #print(self.image_paths[0:5])
  

    # Normalising the images for resnet18 ------------------------------
    # The inference transforms are available at 
    # ResNet18_Weights.IMAGENET1K_V1.transforms and perform the
    # following preprocessing operations: Accepts PIL.Image, batched 
    # (B, C, H, W) and single (C, H, W) image torch.Tensor objects. 
    # The images are resized to resize_size=[256] using 
    # interpolation=InterpolationMode.BILINEAR, 
    # followed by a central crop of crop_size=[224]. 
    # Finally the values are first rescaled to [0.0, 1.0] 
    # and then normalized using mean=[0.485, 0.456, 0.406] 
    # and std=[0.229, 0.224, 0.225]. ------------------------------------
    self.resnet_normalisations = transforms.Compose([
      transforms.Resize(256, interpolation = 
      transforms.InterpolationMode.BILINEAR),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Final augmentations as recommended by Perez, et al. (2018) ------
    self.augmentations = transforms.RandomApply([
      transforms.ToPILImage(),
      transforms.RandomCrop(size = (224,224)),
      transforms.RandomAffine(degrees=90,
                              scale=(0.8, 1.2),
                              shear=20),
      transforms.RandomHorizontalFlip(),
      transforms.RandomVerticalFlip(),
      #transforms.ColorJitter(brightness=(0.7,1.3), contrast=(0.7,1.3),
      #                       saturation=(0.7,1.3), hue=(-0.1,0.1)),
      transforms.ToTensor()
    ],p=0.5)

    # fivecrop challenge 1e --------------------------------------------
    #self.fivecrops = transforms.Compose([
    #  transforms.Resize((400, 600)),
    #  transforms.FiveCrop((200, 300))
      
    #])

  # returns length of dataset ------------------------------------------
  def __len__(self):
    return len(self.labels)

  
  # opens the image using image_path, gets the label for the image
  # returns the image and the label. ----------------------------------
  def __getitem__(self, index):
    image_path = self.image_paths[index]
    label = self.labels[index]

    image = Image.open(image_path)

    # Normalisation for resnet
    inputs = self.resnet_normalisations(image)

    # Augmentations
    if self.augment:
      inputs = self.augmentations(inputs)    

    # Shape checking
    # print(inputs.shape)
    
    return inputs, label

    #fivecrop challenge ------------------------------------------------
    #crops = self.fivecrops(image)

    #five_crop_tensors = []

    #for crop in crops:
    #  crop_tensor = torch.Tensor(np.array(crop))
    #  crop_tensor = crop_tensor.permute(2,0,1) 
    #  five_crop_tensors.append(crop_tensor)
    
    #inputs = torch.stack(five_crop_tensors, dim=0)

    # Combine the 5 crops and 3 channels in to 15 channels
    #inputs = inputs.view(-1, 200, 300)
    #print(inputs.shape)

    #return inputs, label



# TODO Task 1e - Add augment flag to LesionDataset, so the __init__ function
#                now look like this:
#                   def __init__(self, img_dir, labels_fname, augment=False):
#



#class LesionDataset(torch.utils.data.Dataset):
#    pass
