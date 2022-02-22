import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


imgSize = 224

class ResNet(nn.Module):
    def __init__(self, num_classes=14, capacity=16, channel=3, N = 256):
        super(ResNet, self).__init__()
        self.extractor = models.resnet18(pretrained=True)
#         self.extractor = models.resnext50_32x4d(pretrained=True)
        self.extractor_dim = self.extractor.fc.in_features
        self.out_classes = num_classes
        self.extractor.fc = nn.Identity()

        self.proj1 = nn.Linear(self.extractor_dim, N) 
        self.proj2 = nn.Linear(self.extractor_dim, N) 
        self.proj3 = nn.Linear(self.extractor_dim, N) 
        self.proj4 = nn.Linear(self.extractor_dim, N)
        self.proj5 = nn.Linear(self.extractor_dim, N) 
        self.proj6 = nn.Linear(self.extractor_dim, N) 
        self.proj7 = nn.Linear(self.extractor_dim, N) 
        self.proj8 = nn.Linear(self.extractor_dim, N) 
        self.proj9 = nn.Linear(self.extractor_dim, N) 
        self.proj10 = nn.Linear(self.extractor_dim, N) 
        self.proj11 = nn.Linear(self.extractor_dim, N) 
        self.proj12 = nn.Linear(self.extractor_dim, N) 
        self.proj13 = nn.Linear(self.extractor_dim, N) 
        self.proj14 = nn.Linear(self.extractor_dim, N)

        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(N, 1) #class 1
        self.fc2 = nn.Linear(N, 1) #class 2
        self.fc3 = nn.Linear(N, 1) #class 3
        self.fc4 = nn.Linear(N, 1) #class 4
        self.fc5 = nn.Linear(N, 1) #class 5
        self.fc6 = nn.Linear(N, 1) #class 6
        self.fc7 = nn.Linear(N, 1) #class 7
        self.fc8 = nn.Linear(N, 1) #class 8
        self.fc9 = nn.Linear(N, 1) #class 9
        self.fc10 = nn.Linear(N, 1) #class 10
        self.fc11 = nn.Linear(N, 1) #class 11
        self.fc12 = nn.Linear(N, 1) #class 12
        self.fc13 = nn.Linear(N, 1) #class 13
        self.fc14 = nn.Linear(N, 1) #class 14
        self.sigm = nn.Sigmoid()


    def forward(self, x):
        #first, process images    
        out = self.extractor(x)

        #out1 - class
        out1 = self.relu(self.proj1(out))
        out_feat1 = out1
        
        #out2 - class
        out2 = self.relu(self.proj2(out))
        out_feat2 = out2
        
        #out3 - class
        out3 = self.relu(self.proj3(out))
        out_feat3 = out3
        
        #out4 - class
        out4 = self.relu(self.proj4(out))
        out_feat4 = out4
        
        #out5 - class
        out5 = self.relu(self.proj5(out))
        out_feat5 = out5
          
        #out6 - class
        out6 = self.relu(self.proj6(out))
        out_feat6 = out6
        
        #out7 - class
        out7 = self.relu(self.proj7(out))
        out_feat7 = out7
        
        #out8 - class
        out8 = self.relu(self.proj8(out))
        out_feat8 = out8
        
        #out9 - class
        out9 = self.relu(self.proj9(out))
        out_feat9 = out9
        
        #out10 - class
        out10 = self.relu(self.proj10(out))
        out_feat10 = out10
        
        #out11 - class
        out11 = self.relu(self.proj11(out))
        out_feat11 = out11
        
        #out12 - class
        out12 = self.relu(self.proj12(out))
        out_feat12 = out12
        
        #out13 - class
        out13 = self.relu(self.proj13(out))
        out_feat13 = out13
        
        #out14 - class
        out14 = self.relu(self.proj14(out))
        out_feat14 = out14
        
        
        out1 = self.sigm(self.fc1(out1))
        out2 = self.sigm(self.fc2(out2))
        out3 = self.sigm(self.fc3(out3))
        out4 = self.sigm(self.fc4(out4))
        out5 = self.sigm(self.fc5(out5))
        out6 = self.sigm(self.fc6(out6))
        out7 = self.sigm(self.fc7(out7))
        out8 = self.sigm(self.fc8(out8))
        out9 = self.sigm(self.fc9(out9))
        out10 = self.sigm(self.fc10(out10))
        out11 = self.sigm(self.fc11(out11))
        out12 = self.sigm(self.fc12(out12))        
        out13 = self.sigm(self.fc13(out13))
        out14 = self.sigm(self.fc14(out14))

        
            
        return (out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14), (out_feat1, out_feat2, out_feat3, out_feat4, out_feat5, out_feat6, out_feat7, out_feat8, out_feat9, out_feat10, out_feat11, out_feat12, out_feat13, out_feat14)       
        
