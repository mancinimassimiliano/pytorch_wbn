import torch
import wbn_layers
import torch.nn as nn
import torch.nn.functional as F

# Define a model using WBN2d
class Model2d(nn.Module):
    def __init__(self):
        super(Model2d,self).__init__()
        self.layer1 = nn.Conv2d(20,100,1)
        self.wbn = wbn_layers.WBN2d(100,k=10, affine=True)
        self.layer2 = nn.Linear(100,5)
        self.compute_weights = nn.Linear(20,10)
        self.compute_weights.weight.data.fill_(0.001)
        self.compute_weights.bias.data.fill_(0.0)
        
    def forward(self,x):
        w = self.compute_weights(x.view(x.shape[0],-1))
        x = self.layer1(x)
        x = self.wbn(x,F.softmax(w,dim=-1))
        x = x.view(x.shape[0],-1)
        x = self.layer2(F.relu(x))
        return x

# Define a model using WBN1d
class Model1d(nn.Module):
    def __init__(self):
        super(Model1d,self).__init__()
        self.layer1 = nn.Linear(20,100)
        self.wbn = wbn_layers.WBN1d(100,k=10, affine=True)
        self.layer2 = nn.Linear(100,5)
        self.compute_weights = nn.Linear(20,10)
        self.compute_weights.weight.data.fill_(0.001)
        self.compute_weights.bias.data.fill_(0.0)
        
    def forward(self,x):
        w = self.compute_weights(x)
        x = self.layer1(x)
        x = self.wbn(x,F.softmax(w,dim=-1))
        x = self.layer2(F.relu(x))
        return x

# Define a model using WBN1d
class ModelStd(nn.Module):
    def __init__(self):
        super(ModelStd,self).__init__()
        self.layer1 = nn.Linear(20,100)
        self.wbn = wbn_layers.WBN(100,affine=True)
        self.layer2 = nn.Linear(100,5)
        self.compute_weights = nn.Linear(20,1)
        self.compute_weights.weight.data.fill_(0.001)
        self.compute_weights.bias.data.fill_(0.0)
        
    def forward(self,x):
        w = self.compute_weights(x)
        x = self.layer1(x)
        x = self.wbn(x,F.softmax(w,dim=0))
        x = self.layer2(F.relu(x))
        return x
    
# Define a model using both
class MultiModel(nn.Module):
    def __init__(self):
        super(MultiModel,self).__init__()
        self.layer1 = nn.Conv2d(20,100,3,3)
        self.wbn1 = wbn_layers.WBN2d(100,k=10, affine=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.layer2 = nn.Linear(100,30)
        self.wbn2 = wbn_layers.WBN1d(30,k=10, affine=True)
        self.layer3 = nn.Linear(30,5)
        self.compute_weights = nn.Linear(500,10)
        self.compute_weights.weight.data.fill_(0.1)
        self.compute_weights.bias.data.fill_(0.0)
        
    def forward(self,x):
        w = F.softmax(self.compute_weights(x.view(x.shape[0],-1)),dim=-1)
        x = self.layer1(x)
        x = self.wbn1(x,w)
        x = self.avgpool(x).view(x.shape[0],-1)
        x = self.layer2(F.relu(x))
        x = self.wbn2(x,w)
        x = self.layer3(F.relu(x))
        return x


# Test WBN1d grad_flow
x = torch.FloatTensor(128,20).uniform_().to('cuda')
net = ModelStd().to('cuda')
out=net(x)
loss = out.sum()
loss.backward()
assert net.layer1.weight.grad is not None
print('WBN: OK')

    
# Test WBN1d grad_flow
x = torch.FloatTensor(128,20).uniform_().to('cuda')
net = Model1d().to('cuda')
out=net(x)
loss = out.sum()
loss.backward()
assert net.layer1.weight.grad is not None
print('WBN1d: OK')

# Test WBN2d grad_flow
x = torch.FloatTensor(128,20,1,1).uniform_().to('cuda')
net = Model2d().to('cuda')
out=net(x)
loss = out.sum()
loss.backward()
assert net.layer1.weight.grad is not None
print('WBN2d: OK')

# Test multiples WBNs grad_flow
x = torch.FloatTensor(128,20,5,5).uniform_().to('cuda')
net = MultiModel().to('cuda')
out=net(x)
loss = out.sum()
loss.backward()
assert net.layer1.weight.grad is not None
print('Multiple WBNs: OK')
print()

print('All checks have been passed!')
