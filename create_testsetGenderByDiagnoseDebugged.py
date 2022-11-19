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
#NB! Use high resolution to match Vajira's model (it doesn't work with low resolutions...)
#To change resolution, change sampling_rate to 100
sampling_rate=500
print('Reading the data...')
# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)
#Include only the 8 leads that contain information
# (I, II, V1-V6):
X_lead8 = X[:,:,[0,1,6,7,8,9,10,11]]
print('Shape of the X values:',X_lead8.shape)
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
print('Overall sex distribution:')
print(Y['sex'].value_counts())

#Detecting patients with 0 or more than 1 diagnose
print('Detecting patients with 0 or more than 1 diagnose:')
weird_diagnoses = []
index_list_remove = []
for i in range(Y.shape[0]):
    if (len(Y.iloc[i,-1])!=1) or (len(Y.iloc[i,-1])==0):
        #print(Y.iloc[i,-1])
        weird_diagnoses.append(Y.iloc[i,-1])
        index_list_remove.append(i)
print('There are',len(weird_diagnoses),'patients with either 0 or >1 diagnose')

print('The indexes to remove:',index_list_remove[:20])
#Must reset the index of Y to filter out patients with >1 diagnose
#Since Y.drop uses the index column
print(Y.head())
Y =Y.reset_index()
print(Y.head())
#Drop all rows where the patient has more than 1 diagnose
Y_reduced = Y.drop(index_list_remove, axis = 0)
print('Shape of reduced Y data:', Y_reduced.shape)
X_reduced = np.delete(X_lead8,index_list_remove,axis = 0)

#Must reset the index after created Y_reduced:
Y_reduced = Y_reduced.reset_index()
print('Reduced shape of X-data:', X_reduced.shape)
#Extract the diagnose from the list: 
#https://stackoverflow.com/questions/53218931/how-to-unnest-explode-a-column-in-a-pandas-dataframe-into-multiple-rows/53219045#53219045
Y_reduced = Y_reduced.join(pd.DataFrame(Y_reduced.diagnostic_superclass.tolist(),index=Y_reduced.index).add_prefix('Diagnose_'))
#Remove rows with missing values
Y_reduced = Y_reduced.dropna(subset = ['Diagnose_0'])
print('After removing missing rows:', Y_reduced.head(30))
print(Y_reduced.shape)
remaining_rows = Y_reduced.index.tolist()
print('Number of remaining rows:',len(remaining_rows))
#Remove the same rows for the X data:
X_reduced = X_reduced[remaining_rows]
print('New shape of X data:', X_reduced.shape)

#Factorize the columns in Y as diagnoses
factor = pd.factorize(Y_reduced['Diagnose_0'])
#Store the factorized columns as diagnoses:
#https://www.codementor.io/@agarrahul01/multiclass-classification-using-random-forest-on-scikit-learn-library-hkk4lwawu
Y_reduced['diagnose'] = factor[0]
print('Remaining patients in the data set with exactly 1 diagnose:', Y_reduced.shape[0])


# Only need the test set
test_fold = 10

X_test = X_reduced[np.where(Y_reduced.strat_fold == test_fold)]
y_test = Y_reduced[Y_reduced.strat_fold == test_fold]

print('For the test set:', y_test.iloc[:20,-2:])

#Select only NORM
#For selecting MI patients, replace 'NORM' with 'MI' in the four lines of code below
print('Ids before changing y_test:', np.where(y_test.Diagnose_0 =='NORM'))
#The order here is important to ensure that not only the NN first rows are taken from X_test
X_test = X_test[np.where(y_test.Diagnose_0 =='NORM')]
y_test = y_test[y_test.Diagnose_0 == 'NORM']
print('Ids AFTER changing y_test:', np.where(y_test.Diagnose_0 =='NORM'))


print('Diagnose distribution:', y_test['Diagnose_0'].value_counts())
print('Gender distribution:', y_test['sex'].value_counts())
#Select the gender column for predictions:
print(y_test[:10])
y_test = y_test.sex
print(X_test[:10])
print(y_test[:10])

print('Shape of test sets:',X_test.shape)

X_test = X_test.reshape(X_test.shape[0],40000)

print(y_test.value_counts())

print('Writing files to csv...')
pd.DataFrame(X_test).to_csv('X_testGenderNORM.csv', index_label=None)
#Also write y-data to csv:
y_test.to_csv('y_test_GenderNORM.csv', index_label=None)
