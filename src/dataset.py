import os
from PIL import Image

import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset
    
chest_transform = transforms.Compose([transforms.Resize(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5066, 0.5066, 0.5066], std=[0.2284, 0.2284, 0.2284])
                                   ])

classes = ['no finding','enlarged cardiomediastinum', 'cardiomegaly', 'lung lesion', 'lung opacity', 'edema', 
           'consolidation', 'pneumonia', 'atelectasis', 'pneumothorax', 'pleural effusion', 'pleural other',
           'fracture','support devices']



    
class ChestDataset(Dataset):
    def __init__(self, data, phase="train", transform=chest_transform):
        self.data = data
        self.phase = phase
        self.transform = transform
        
    def get_class_labels(self, labels):
        s = ""
        labels = list(labels)
        for i in range(len(labels)):
            if labels[i] == 1:
                if s != "":
                    s += ","
                s += classes[i]
                    
        return s    
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        fpath, labels = self.data[idx]
        img = Image.open(fpath).convert("RGB")
        if self.transform!=None:
            img = self.transform(img)  
        if self.phase != "rank":
            return img, torch.LongTensor(list(labels)), self.get_class_labels(labels)   
        else:    
            return img, torch.LongTensor(list(labels)), self.get_class_labels(labels), fpath   
