import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset,  DataLoader
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.collections import LineCollection

from pytorch_grad_cam import GradCAM
from vajiraGAN_original import Pulse2pulseDiscriminator

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#To look at MI ECGs, change X_TEST_PATH and Y_TEST_PATH 
# to corresponding files after running 
# 'create_testsetGenderByDiagnoseDebugged.py'

#Inspecting 'Normal' ECGs:
X_TEST_PATH = './X_testGenderNORM.csv'
Y_TEST_PATH = './y_test_GenderNORM.csv'

#Define how to test the model:
def create_heatmaps(model, dataloader, cam_object):
    #Get the predicted probabilities:
    all_probs = []
    true_gender = []
    all_titles = []
    all_imageNumbers = []
    for i, (inputs, y_true) in enumerate(dataloader):
        print('Round number',str(i+1))
        inputs = inputs.to(DEVICE)
        y_true = y_true.to(DEVICE)
        y_pred = model(inputs.float())
        
        
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu()
        #Also get the predicted probability:
        predicted_probs = torch.sigmoid(y_pred)
        y_pred = y_pred.numpy()
        
        #Convert to 0 or 1, using threshold of 0
        predicted_vals = y_pred > 0
        #Print the predictions:
        #print('Predicted:', predicted_vals)
        #print('Target:', y_true)
        for j in range(len(y_pred)):
            saliency = cam_object(input_tensor=inputs[j].unsqueeze(0).float(), targets=None)
            #print('Selected image label:', predicted_vals[j])
            #print('True label:', y_true[j])
            # If predicted and true labels match, we plot the heatmaps:
            if (y_true[j] == 0 and predicted_vals[j] == False) or (y_true[j] == 1 and predicted_vals[j] == True):
                if y_true[j] == 0:
                    title_string = 'NORM_male_' + str(j) + '_' + str(i) 
                    figure_string = 'NORM_male_ECGoverlaid_V2/heatmap_NORM_male_V2_' + str(j)+ '_' + str(i) +'.png'
                else:
                    title_string = 'NORM_female_' + str(j) + '_' + str(i) 
                    figure_string = 'NORM_female_ECGoverlaid_V2/heatmap_NORM_female_V2_' + str(j)+ '_' + str(i) + '.png'
                
                #Add the predicted probability:
                all_probs.append(predicted_probs[j][0].item())
                #Add the true gender:
                true_gender.append(y_true[j][0])
                #Add the title_string:
                all_titles.append(title_string)
                #Add the image number:
                image_number = i * 30 + j + 1
                all_imageNumbers.append(image_number)
                
                #Want to overlay ECG onto the heatmap:
                data_points = np.zeros((inputs[j].shape[1], 1, 2))
                #Select lead number 2 (=II) for plotting signals in the attribution map:
                # The 8 included leads are (I, II, V1-V6)
                for row_index, point in enumerate(inputs[j][1,:]):
                    data_points[ row_index,0, 0 ] = row_index
                    data_points[ row_index,0, 1 ] = point
                segments = np.hstack([data_points[:-1], data_points[1:]])
                coll = LineCollection(segments, colors=[ [ 0, 0, 0 ] ] * len(segments), linewidths=(1.3)) 
                
                fig, ax = plt.subplots(1, 1, figsize=(14,6))
                ax.add_collection(coll)
                ax.autoscale_view()
                ax.title.set_text(title_string)

                saliency = np.transpose(saliency, (1, 2, 0))
                #Convert saliency to numpy and then to uint8 for RGB colors:
                saliency = cv2.applyColorMap((saliency * 255).astype("uint8"), cv2.COLORMAP_JET)
                saliency = cv2.cvtColor(saliency, cv2.COLOR_BGR2RGB)
                saliency_colors = saliency / 255
                saliency_colors = np.mean(saliency_colors, axis=0)
                for c_index, color in enumerate(saliency_colors):
                    ax.axvspan(c_index, c_index+1, facecolor=color)
                plt.tight_layout()
                plt.savefig(figure_string)
                plt.close()
                
                
            
    #Create an empty df and add the predicted probas, ground truth, image names, etc
    proba_df = pd.DataFrame(all_probs, columns=['Predicted_proba'])
    proba_df['Image_number'] = all_imageNumbers
    proba_df['Image_title'] = all_titles
    proba_df['Gender'] = true_gender
    return proba_df


def run():

    X_test = pd.read_csv(X_TEST_PATH, index_col='Unnamed: 0')
    y_test = pd.read_csv(Y_TEST_PATH, index_col='ecg_id')
    print('Shape of NORM testset:', X_test.shape)

    #Convert to numpy array and use correct shape:
    X_test = X_test.to_numpy(dtype=np.float64).reshape(X_test.shape[0],5000,8)
    #Must then transpose to get the correct order:
    X_test = np.transpose(X_test,(0,2,1))
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test.to_numpy(dtype=np.float64))

    #Normalize by dividing by 6000...
    X_test_tensor = X_test_tensor / 6000

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False, num_workers=0)

    print('Loading the pretrained model...')
    #NB! The model size should be 50!
    model = Pulse2pulseDiscriminator(model_size=50, shift_factor=0)
    #Load the checkpoints:
    checkpoint_path = 'trained_model_Gender5000Samples1000epochs.pt'
    chkpnt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(chkpnt)
    model.eval()

    cam = GradCAM(model=model,
                     target_layers=[model.conv6],
                     use_cuda=False)
    my_probaDF = create_heatmaps(model, test_loader,cam)
    #Save df as csv:
    my_probaDF.to_csv('NORM_predictedProbasDF.csv', index_label=None)
if __name__ == "__main__":
    run()