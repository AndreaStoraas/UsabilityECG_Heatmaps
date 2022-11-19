import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import TensorDataset,  DataLoader
from sklearn import metrics
import pandas as pd
import numpy as np

from vajiraGAN_original import Pulse2pulseDiscriminator
print('Reading in the test data...')

# For testing on only NORM/MI patients, change the paths for X_test and y_test
# so that they match with the respective files
# acquired by running 'create_testsetGenderByDiagnoseDebugged.py'
#Testing on the entire test set:
X_test = pd.read_csv('./X_test_gender8leads500records.csv', index_col='Unnamed: 0')
y_test = pd.read_csv('./y_test_gender.csv', index_col='ecg_id')

print(y_test.head())
#Convert to numpy array and use correct shape:
X_test = X_test.to_numpy(dtype=np.float64).reshape(X_test.shape[0],5000,8)
#Must then transpose to get the correct order:
X_test = np.transpose(X_test,(0,2,1))

print(X_test)
print(X_test.shape)
print(y_test.shape)

print('Converting the test set to tensors...')
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test.to_numpy(dtype=np.float64))

#Normalize by dividing by 6000...
X_test_tensor = X_test_tensor / 6000

#Create the test dataset:
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

#Define how to test the model:
def test_model(model, dataloader):
    all_predicted = []
    #Uncomment to also get the predicted probablities:
    #all_probs = []
    model.eval()

    running_acc = 0.0
    running_fscore = 0.0
    with torch.set_grad_enabled(False):
        for param in model.parameters():
           param.requires_grad = False
           print(param)
        for i, (inputs, y_true) in enumerate(dataloader):
            print(i)
            inputs = inputs.to(DEVICE)
            y_true = y_true.to(DEVICE)
            y_pred = model(inputs.float())
            ##Uncomment to also get the predicted probablities::
            #predicted_probs = torch.sigmoid(y_pred)
            
            y_true = y_true.detach().cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy()
            #Convert to 0 or 1, using threshold of 0
            predicted_vals = y_pred > 0
            
            #Print the predictions:
            print('Predicted:', predicted_vals)
            print('Target:', y_true)

            running_acc += metrics.accuracy_score(y_true, predicted_vals) 
            running_fscore += metrics.f1_score(y_true, predicted_vals)
            
            all_predicted.append(predicted_vals)
            #Uncomment to also get the predicted probablities:
            #all_probs.append(predicted_probs)
            print("Running ACC: %.4f" % (metrics.accuracy_score(y_true, predicted_vals)))
    
    mean_acc = running_acc / len(dataloader)
    mean_fscore = running_fscore / len(dataloader)
    #Flatten the list
    all_predicted = [a.squeeze().tolist() for a in all_predicted] 
    #Uncomment to also get the predicted probablities:
    #all_probs = [a.squeeze().tolist() for a in all_probs]        
    print("Overall Acc on test set: %.4f" % mean_acc)
    print('overall F1-score on test set:', mean_fscore)
    #Uncomment to also get the predicted probablities:
    #return all_predicted, all_probs
    return all_predicted

#Define the device:
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Load the pretrained model:
print('Loading the pretrained model...')
#NB! The model size should be 50!
model = Pulse2pulseDiscriminator(model_size=50, shift_factor=0)
#Load the checkpoints:
print('Testing on the model trained on 5000 samples...')
checkpoint_path = './trained_model_Gender5000Samples1000epochs.pt'
chkpnt = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(chkpnt)

model.to(DEVICE)

#Uncomment to also get the predicted probablities:
#predicted_values, predicted_probabilities = test_model(model, test_loader)
predicted_values = test_model(model, test_loader)

#Flatten the list...
predicted_values = [item for sublist in predicted_values for item in sublist]
print('Number of predictions:', len(predicted_values))
print(metrics.confusion_matrix(y_test, predicted_values))

print(metrics.classification_report(y_test, predicted_values))

#Uncomment to also get the predicted probablities:
#Flatten the list for predicted probabilities:
#Get the last predicted probability:
#predicted_probabilities = predicted_probabilities[:-1]
#predicted_probabilities = [item for sublist in predicted_probabilities for item in sublist]
#print('Predicted probabilities:', predicted_probabilities)
#print('Predicted classes:', predicted_values)