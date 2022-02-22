import os
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import *
from src.DEML14 import ResNet
from src.dataset import ChestDataset


trainlist = list(np.load("/path/to/chexpert_train.npy", allow_pickle=True))
vallist = list(np.load("/path/to/chexpert_test.npy", allow_pickle=True))
print(len(trainlist), len(vallist))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


batch_size = 64
num_workers = 8


imratio = np.array([22012, 7049, 17743, 5589, 52889, 35478, 8678, 3894, 23540, 15002, 55566, 1976, 5987, 73499])/133102

trainset = ChestDataset(trainlist, "train")
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valset = ChestDataset(vallist, "test")
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

warmup_epochs = 6
best_loss = 1e5

N = 256
n_classes = 14
model = ResNet(n_classes, N=N )
model = model.to(device)

CLS_criterion = nn.BCELoss()


codebook_classwise_num = 1000
codebook = np.zeros((codebook_classwise_num*n_classes, N))
codebook = torch.from_numpy(codebook)
codebook = codebook.to(device)

learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)

#update codebook
counts = [i*codebook_classwise_num for i in range(n_classes)]

classes = ['no finding','enlarged cardiomediastinum', 'cardiomegaly', 'lung lesion', 'lung opacity', 'edema', 
           'consolidation', 'pneumonia', 'atelectasis', 'pneumothorax', 'pleural effusion', 'pleural other',
           'fracture','support devices']
classes_dict = {'no finding' : 0, 'enlarged cardiomediastinum': 1, 'cardiomegaly': 2, 'lung lesion': 3, 'lung opacity': 4, 'edema': 5, 
           'consolidation': 6, 'pneumonia': 7, 'atelectasis': 8, 'pneumothorax': 9, 'pleural effusion': 10, 'pleural other': 11,
           'fracture': 12, 'support devices': 13}


def pms(outfeat, targets, idx, c, imratio=imratio, R=1, tau = 1, cosine=True):
    # c: class
    pos_outfeat = []
    neg_outfeat = []
    c = c.unsqueeze(0)
    for i in range(outfeat.size(0)):
        if targets[i][idx] == 1:
            pos_outfeat.append(outfeat[i])
        else:
            neg_outfeat.append(outfeat[i])
    if len(pos_outfeat):        
        pos_outfeat = torch.vstack(pos_outfeat)
    if len(neg_outfeat):     
        neg_outfeat = torch.vstack(neg_outfeat)  
    pms_loss = 0.0
    if cosine: 
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        if len(pos_outfeat):
            pms_loss = torch.mean(cos(c, pos_outfeat)/tau)*(1-imratio[idx])
        if len(neg_outfeat):
            tcos = cos(c, neg_outfeat)/tau
            pms_loss += torch.mean(torch.nn.functional.relu(R-tcos))*imratio[idx]
    else:
        if len(pos_outfeat): 
            dist = torch.sum((pos_outfeat - c) ** 2, dim=1)
            pms_loss = -(1-imratio[idx])*torch.mean(dist)
        if len(neg_outfeat):
            tdist = torch.sum((neg_outfeat - c) ** 2, dim=1)
            pms_loss += -imratio[idx]*torch.mean(torch.nn.functional.relu(R-tdist))
    return pms_loss  
  
  
  cs = torch.load('./weights/cs_chest.pt', map_location=device) # proxy centers
  
warmup_epochs=5
for epoch in range(warmup_epochs):
    #train phase
    running_loss = 0.0
    model.train()
    train_pred = []
    train_true = []
    test_pred = []
    test_true = []
    for imgs, targets, class_labels in tqdm(trainloader):
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        preds, out_feat = model(imgs)
        combined_preds = torch.stack(preds, dim=1).squeeze(2)
        train_pred.append(combined_preds.cpu().detach().numpy())
        train_true.append(targets.cpu().detach().numpy())
        pms_loss = 0.0
        cls_loss=CLS_criterion(combined_preds, targets.type(torch.float))
        for j in range(0, n_classes):
            pms_loss += pms(out_feat[j], targets, j, cs[j])
            indices = torch.nonzero(targets[:,j]).squeeze(1)
            selected_feat = torch.index_select(out_feat[j], 0, indices)
            if counts[j]+selected_feat.size(0) < (j+1)*codebook_classwise_num:
                codebook[counts[j]:counts[j]+selected_feat.size(0)]=selected_feat
                counts[j] += selected_feat.size(0)
            else:
                t = (j+1)*codebook_classwise_num - counts[j]
                codebook[counts[j]:(j+1)*codebook_classwise_num]=selected_feat[0:t]
                codebook[j*codebook_classwise_num:j*codebook_classwise_num+selected_feat.size(0)-t]=selected_feat[t:]
                
        loss = 0.25*cls_loss -  0.75*(pms_loss)/n_classes
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)


    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    train_auc_mean =  roc_auc_score(train_true, train_pred)         
        
    train_epoch_loss = running_loss / len(trainset)
    
    #test phase
    running_loss = 0.0
    model.eval()
    for imgs, targets, shape_labels in tqdm(valloader):
        imgs, targets = imgs.to(device), targets.to(device)
        preds, out_feat = model(imgs)
        combined_preds = torch.stack(preds, dim=1).squeeze(2)
        test_pred.append(combined_preds.cpu().detach().numpy())
        test_true.append(targets.cpu().detach().numpy())
        
        cls_loss=CLS_criterion(combined_preds, targets.type(torch.float))
        pms_loss = 0.0
        for j in range(0, n_classes):
            pms_loss += pms(out_feat[j], targets, j, cs[j])
        loss = 0.25*cls_loss  - 0.75*(kernel_loss)/n_classes
        running_loss += loss.item() * imgs.size(0)
    
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
          
    test_epoch_loss = running_loss / len(valset)
    print("epoch:{:2d}  train: loss:{:.3f} val: loss:{:.3f}".format(epoch, train_epoch_loss, test_epoch_loss))
    ways = ['micro', 'macro', 'weighted']
    print("None: ", roc_auc_score(test_true,  np.array(test_pred>0.5,dtype=int), average=None, multi_class='ovr'))
    print(ways[0], roc_auc_score(test_true,  np.array(test_pred>0.5,dtype=int), average=ways[0], multi_class='ovr'))
    print(ways[1], roc_auc_score(test_true,  np.array(test_pred>0.5,dtype=int), average=ways[1], multi_class='ovr'))
    print(ways[2], roc_auc_score(test_true,  np.array(test_pred>0.5,dtype=int), average=ways[2], multi_class='ovr'))
    
    if test_epoch_loss < best_loss:
        best_loss = test_epoch_loss 
        save_path = "./weights/PMS_chest_best.pt"
        torch.save({'model_state_dict': model.state_dict()}, save_path)            
  
  
  
  
 
  
  
  
