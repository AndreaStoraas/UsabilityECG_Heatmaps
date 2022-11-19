import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.utils.data
#from torchvision import transforms, models
from torch.utils.data import TensorDataset,  DataLoader
import copy
from sklearn import metrics
#import wfdb
#import ast
import pandas as pd
import numpy as np

from vajiraGAN_original import Pulse2pulseDiscriminator

#Including 8 leads...
print('Reading the data files...')
X_train = pd.read_csv('./X_train_gender8leads500records.csv', index_col='Unnamed: 0')
X_val = pd.read_csv('./X_val_gender8leads500records.csv', index_col='Unnamed: 0')
y_train = pd.read_csv('./y_train_gender.csv', index_col='ecg_id')
y_val = pd.read_csv('./y_val_gender.csv', index_col='ecg_id')


print(X_train.shape)
#Concat the df's before selecting random samples:
#NB! Must reset the index to avoid extra rows containing NaNs!
full_train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis = 1)
full_val = pd.concat([X_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis = 1)

#Select only 5,000 rows randomly from training set:
full_train = full_train.sample(n=5000, random_state=42)
#Use 200 rows for validation:
full_val = full_val.sample(n=200, random_state=42)

print(full_train.head())

#Split back into X and y again:
X_trainSmall = full_train.iloc[:,:-1]
y_trainSmall = full_train.iloc[:,-1]
X_valSmall = full_val.iloc[:,:-1]
y_valSmall = full_val.iloc[:,-1]

print('Shape of training set:')
print(X_trainSmall.shape)
print('Shape of y_val:')
print(y_valSmall.shape)

#Since not all rows are used:
#Since 500 sampling rate instead of 100, reshape to 1000, 5000, 8 etc
X_trainSmall = X_trainSmall.to_numpy(dtype=np.float64).reshape(5000,5000,8)
X_valSmall = X_valSmall.to_numpy(dtype=np.float64).reshape(200,5000,8)

#Must then transpose to get input from [n, 1000, 8] to [n,8,1000]:
X_trainSmall = np.transpose(X_trainSmall,(0,2,1))
X_valSmall = np.transpose(X_valSmall,(0,2,1))

print('Males/females in training set n=5000:', y_trainSmall.value_counts())
print('Males/females in validation set n=200:', y_valSmall.value_counts())
print('Converting data sets to tensors:')
#Convert the data sets to tensors: 
X_trainSmall = torch.tensor(X_trainSmall)
y_trainSmall = torch.tensor(y_trainSmall.to_numpy(dtype=np.float64))

X_valSmall = torch.tensor(X_valSmall)
y_valSmall = torch.tensor(y_valSmall.to_numpy(dtype=np.float64))


#Normalize the values as Vajira did (dividing by 6000)
X_trainSmall = X_trainSmall / 6000
X_valSmall = X_valSmall / 6000


#Create the training and testing datasets:
train_dataset = TensorDataset(X_trainSmall, y_trainSmall)
val_dataset = TensorDataset(X_valSmall, y_valSmall)
#Create the dataloaders used to train and test the model: 
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)


#Define how to train the model:
def train(model, dataloaders, optimizer, criterion, n_epochs): 
        
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch_idx in range(n_epochs):

        for phase, dataloader in dataloaders.items():
            
            if phase == "TRAIN":
                model.train()
                #Set the requires_grad = True
                for p in model.parameters():
                    p.requires_grad = True
            else:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
            
            running_loss = 0.0
            running_acc = 0.0

            with torch.set_grad_enabled(phase == "TRAIN"):

                for i, (inputs, y_true) in enumerate(dataloader):
                    #print(i)
                    inputs = inputs.to(DEVICE)
                    y_true = y_true.to(DEVICE)
                    
                    y_pred = model(inputs.float())
                    #Change dimensions of y_true:
                    y_true = y_true.unsqueeze(1)
                    loss = criterion(y_pred, y_true)

                    if phase == "TRAIN":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                    
                    y_true = y_true.detach().cpu().numpy()
                    y_pred = y_pred.detach().cpu().numpy()
                    # Convert to 0 or 1, using threshold of 0
                    predicted_vals = y_pred > 0
                    
                    running_loss += loss.item()
                    running_acc += metrics.accuracy_score(y_true, predicted_vals) 

            mean_loss = running_loss / len(dataloader)
            mean_acc = running_acc / len(dataloader)
            
            if phase == "VALID" and mean_acc > best_acc:
                best_acc = mean_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            print("%s Epoch %i\t Loss: %.4f\t ACC: %.4f" % (phase, epoch_idx, mean_loss, mean_acc))
            
    print("Best val Acc: %.4f" % best_acc)
    model.load_state_dict(best_model_wts)
    return model


#Define the device:
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Type of device:', DEVICE)
#Load the pretrained model:
print('Loading the pretrained model...')
#NB! The model size should be 50!
model = Pulse2pulseDiscriminator(model_size=50)
#Load the checkpoints:
#checkpoint_path = './DiscriminatorWeights/006_WG_RYTHM_10s_normal_only_wave2wave_from_2100_v1.py_epoch:2500.pt'
checkpoint_path = './Checkpointpath/pretrainedModel.pt'

chkpnt = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(chkpnt['netD_state_dict'])

model.to(DEVICE)
#Use same optimizer and hyperparam values as Vajira
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, betas=(0.5,0.9))
#Since binary classification, use BCEWithLogitsLoss:
criterion = nn.BCEWithLogitsLoss()

#Train the model for 1000 epochs: 
print('Starting to train the model...')
best_model = train(model, {'TRAIN': train_loader, 'VALID': val_loader}, optimizer,criterion, n_epochs=1000)

#Save the model:
model_save_path = './trained_model_Gender5000Samples1000epochs.pt'
torch.save(best_model.state_dict(), model_save_path)
