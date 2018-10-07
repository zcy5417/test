# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import datasets,transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
import os

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainset=datasets.MNIST(root='.',train=True,download=False,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))
                                ]))
testset=datasets.MNIST(root='.',train=False,download=False,
                       transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))
                                ]))
train_loader=data.DataLoader(trainset,batch_size=64,shuffle=True,num_workers=4)
test_loader=data.DataLoader(testset,batch_size=64,shuffle=False,num_workers=4)

class simpleNN(nn.Module):
    def __init__(self):
        super(simpleNN,self).__init__()
        self.conv1=nn.Conv2d(1,10,7)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(10,20,5)
        self.fc=nn.Linear(3*3*20,10)
    
    def forward(self,x):
        out=self.conv1(x)
        out=self.relu(out)
        out=self.maxpool(out)
        out=self.conv2(out)
        out=self.relu(out)
        out=self.maxpool(out)
        out=out.view(-1,3*3*20)
        out=self.fc(out)        
        return out
    
net=simpleNN()
net.to(device)

optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
criterion=nn.CrossEntropyLoss()

def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
#    iters=len(trainset)//64+1
#    pbar=tqdm(range(iters))
#    qq=iter(pbar)
#    with tqdm(total=iters) as pbar:

    for batch,(inputs,labels) in enumerate(train_loader):
        
        inputs,labels=inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
#                progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if batch%1000==0:
            print('epoch',epoch,'acc',correct/total)
            
    #        next(qq)
    #        err=batch
    #        pbar.set_description("Reconstruction loss: %s" %(err))
#            pbar.update(1)
#            pbar.set_description("loss: %s" %(batch)) 
            
                
#from tqdm import tqdm
#from time import sleep
#
#pbar = tqdm(range(300))#进度条
#
#for i in pbar:
#    err = i
#    sleep(1)
#    pbar.set_description("Reconstruction loss: %s" %(err))
            
            
            
    return train_loss
#    print(epoch)
     
best_acc=0
def test():
    global best_acc##
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    loss_list=[]
    with torch.no_grad():
        for batch,(inputs,labels) in enumerate(test_loader):
            inputs,labels=inputs.to(device),labels.to(device)
            outputs=net(inputs)
            loss=criterion(outputs,labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
#                        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            if batch%1000==0:
                print('acc',correct/total)
                # Save checkpoint.
                
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'acc':acc
            }, './checkpoint/ckpt.tar')
        best_acc = acc
        
    return test_loss
            
minloss=float('inf')
wait=0
def earlyStopping(loss,patience=1):
    global minloss
    global wait
    if loss<minloss:
        minloss=loss
        wait=0
    else:
        wait += 1
        if wait >= patience:
            return       
            
epoches=50

epoch_list=[]
train_loss_list=[]
test_loss_list=[]

for epoch in range(epoches):
    train_loss=train(epoch)
    test_loss=test()
    scheduler.step(test_loss)
    
    epoch_list.append(epoch)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    
    plt.plot(epoch_list,train_loss_list,color='blue')
    plt.plot(epoch_list,test_loss_list,color='red')
    plt.pause(0.1)
    plt.savefig("loss_fig.png")
    
    
    stopFlag=earlyStopping(test_loss)
    if(stopFlag==1):
        print('epoch stop',epoch)
        break
##### 
    
#    1.lr_scheduler
#    2.earlystopping
#    3.progressbar
#    4.plot
#
#
#

    


#import  time  
#from progressbar import *  
#  
#total = 1000  
#  
#def dosomework():  
#    time.sleep(0.01)  
#  
#widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),  
#           ' ', ETA(), ' ', FileTransferSpeed()]  
#pbar = ProgressBar(widgets=widgets, maxval=10*total).start()  
#for i in range(total):  
#    # do something  
#    pbar.update(10 * i + 1)  
#    dosomework()  
#pbar.finish()  

    
    
#    
#from tqdm import tqdm,trange
#from time import sleep
##
#pbar = tqdm(range(300))#进度条
##
#for i in pbar:
#    err = i
#    sleep(1)
#    pbar.set_description("Reconstruction loss: %s" %(err))
#
#pbar = tqdm(["a", "b", "c", "d"])
#for char in pbar:
#    pbar.set_description("Processing %s" % char)
        
    
    
#with tqdm(total=100) as pbar:
#    for i in range(10):
#        sleep(0.5)
#        pbar.update(1)
#        pbar.set_description("Reconstruction loss: %s" %(i))