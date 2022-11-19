#Importing libraries:
import pandas as pd
import numpy as np
import wfdb
import ast


#Define function to load the data:
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = './'
#NB! Must use high resolution to match Vajira's model (it doesn't work with low resolutions...)
sampling_rate=500

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)



print('Shape of X before extracting the 8 leads:', X.shape)
print(X[0,:,:])
#Include the 8 leads that contain information
# (I, II, V1-V6):
X_lead8 = X[:,:,[0,1,6,7,8,9,10,11]]


# Split data into train, validation and test
test_fold = 10
val_fold = 9
# Train
X_train = X_lead8[np.where(Y.strat_fold < val_fold)]
#Start with predicting gender: 
y_train = Y[(Y.strat_fold < val_fold)].sex

# Validation
X_val = X_lead8[np.where(Y.strat_fold == val_fold)]
y_val = Y[Y.strat_fold == val_fold].sex
# Test
X_test = X_lead8[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].sex

print('Shape of X train:', X_train.shape)
print('Shape of X val:', X_val.shape)
print('Shape of X test:', X_test.shape)


#Since 8 leads, reshape to 8 * 5000 = 40000
X_train = X_train.reshape(17441,40000)
X_val = X_val.reshape(2193,40000)
X_test = X_test.reshape(2203,40000)

print('Writing files to csv...')

pd.DataFrame(X_train).to_csv('X_train_gender8leads500records.csv', index_label=None)
pd.DataFrame(X_val).to_csv('X_val_gender8leads500records.csv', index_label=None)
pd.DataFrame(X_test).to_csv('X_test_gender8leads500records.csv', index_label=None)

#Also write y-data to csv:
y_train.to_csv('y_train_gender.csv', index_label=None)
y_val.to_csv('y_val_gender.csv', index_label=None)
y_test.to_csv('y_test_gender.csv', index_label=None)
