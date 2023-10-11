from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter # to print to tensorboard
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from Models.resnet import *
from Models import *
# from Models.convnext import *
from torch.cuda.amp import autocast as autocast
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import time

prefix = 'example' #
ls_data = torch.load(f'{prefix}_data.pt')
ls_label = torch.load(f'{prefix}_label.pt')

ModelName = "TraceNet"
num_class = 3
lr=1e-3 
batchsz=6 
epochs=200
num_workers=8
pin_memory=True


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long)) # .unsqueeze(0)

    def __len__(self):
        return len(self.data)


def evalute(modeld, val_dl, topk=1):
    modeld.eval()
    with torch.no_grad():
        correct = 0
        total = len(val_dl.dataset)
        for dx, dy in val_dl:
            dx, dy = dx.cuda(non_blocking=True), dy.cuda(non_blocking=True)
            logits = modeld(dx)
            val_loss = criteon(logits, dy)
            pred = logits.argmax(dim=1)
            correct += torch.eq(pred, dy).float().sum().item()
    return correct / total, val_loss


def evatest(modeld, val_dl, topk=1):
    modeld.eval()
    with torch.no_grad():
        correct = 0
        total = len(val_dl.dataset)

        for dx, dy in val_dl:
            dx, dy = dx.cuda(non_blocking=True), dy.cuda(non_blocking=True)
            logits = modeld(dx)
            vals, indices = logits.max(dim=1, keepdim=True) # ,keepdim=True            
            pred = logits.argmax(dim=1)
            correct += torch.eq(pred,dy).float().sum().item()
    return correct/total


cv_acc=[]     

dirs='./CKPT/'
if not os.path.exists(dirs):
    os.makedirs(dirs)

ou='./Output/'
if not os.path.exists(ou):
    os.makedirs(ou)
    
if not os.path.exists("runs"):
    os.makedirs("runs")

for i in range(1):
    modeld = ResidualNet_one(50, num_class, 'CBAM')
    modeld = nn.DataParallel(modeld)
    modeld = modeld.cuda()
    criteon = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(modeld.parameters(), lr=lr)  # ,weight_decay=1e-3
    print('*'*80)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,verbose=True)
    print('Repeat{}:'.format(i))
    print('^' * 80)   
    
    X_train, X_test, Y_train, Y_test = train_test_split(ls_data, ls_label, stratify=ls_label, random_state=42)
    print("Y_test: ", Y_test)
    best_acc, best_epoch = 0, 0
    global_step = 0
    
    tb_path = f"runs/{ModelName}_BatchSize{batchsz}_LearningRate{lr}_Epoch{epochs}"
    
    writer = SummaryWriter(f'{tb_path}/{ModelName}_Repeat{i}_BatchSize{batchsz}_LearningRate{lr}_Epoch{epochs}')


    for epoch in range(epochs):
        print('Epoch: ', epoch)
        print('Train labels: ', Y_train)
        print('Test labels: ', Y_test)
        train_d = MyDataset(X_train, Y_train)
        val_d = MyDataset(X_test, Y_test)
        train_dl = DataLoader(train_d, batch_size=batchsz, num_workers=num_workers, pin_memory=pin_memory)
        val_dl = DataLoader(val_d, num_workers=num_workers, pin_memory=pin_memory)

        modeld.train()
        for dx,dy in train_dl:
            dx, dy =dx.cuda(non_blocking=True),dy.cuda(non_blocking=True)
            # with autocast():
            logits = modeld(dx)
            loss = criteon(logits, dy)
            print('Train Loss: ' + str(loss))
            writer.add_scalar('Training loss', loss, global_step=global_step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        if epoch % 1 == 0:
            val_acc,val_loss = evalute(modeld, val_dl)
            scheduler.step(val_loss) 
            print('Test Loss: ' + str(val_loss)) 
            print('Test Accuracy: ', val_acc)
            writer.add_scalar('Test loss', val_loss, global_step=global_step)
            writer.add_scalar('Test Accuracy', val_acc, global_step=global_step)
            print(80*'#')
            if val_acc >= best_acc:
                best_epoch = epoch
                best_acc = val_acc
                state_dp = f'./CKPT/{ModelName}-batchsz{batchsz}-lr{lr}-epochs{epochs}.mdl' #.format(batchsz,lr,epochs)
                torch.save(modeld.state_dict(), state_dp)

           
    cv_acc.append(best_acc)
    print('Best Accuracy: ',best_acc,'Best epoch: ',best_epoch)
    modeld.load_state_dict(torch.load(state_dp))
    # modelp.load_state_dict(torch.load(state_pp))
    print(80*'*')
    print('Loaded from CKPT!!!')
    time_start = time.time()  
    test_acc=evatest(modeld, val_dl)
    time_end = time.time()  
    time_sum = (time_end - time_start) / 60  
    print('Test Accuracy: ',test_acc) 

cv_mean=np.mean(cv_acc)
print('Cross-Validation Accuracy',cv_mean)
with open('./Output/{}-CV-batchsz{}-lr{}-epochs{}-result.txt'.format(ModelName,batchsz,lr,epochs), 'a') as f:
    for line in cv_acc:
        f.write(str(line)+'\n')
    f.write('cv_mean:  '+str(cv_mean)+'\n')
    

print(f'time_sum: {time_sum}')
writer.add_text('time_sum', str(time_sum), 0)