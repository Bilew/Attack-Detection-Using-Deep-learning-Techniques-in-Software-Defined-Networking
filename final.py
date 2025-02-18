import os #o/p 
import sys #o/p
import subprocess
import subprocess
process = subprocess.Popen(['echo', 'More output'],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
stdout, stderr
from subprocess import run  # i add it to see multiple output python above 3.5
#output = run("pwd", capture_output=True).stdout # i add it to see multiple output
#subprocess.check_output(['ls', '-l'])
#b'total 0\n-rw-r--r--  1 memyself  staff  0 Mar 14 11:04 files\n'

import time  
import os
import math
import sys
import operator
import requests
#load dataset
import csv
#read dataset
import keras as sk
import pandas as pd
import tensorflow as tf
import numpy as np
import sklearn
import seaborn as sn
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
##data preprocessing 
from keras.preprocessing import sequence
from keras.models import Sequential
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
#from tensorflow.keras.utils import to_categorical
#from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
from sklearn.model_selection import train_test_split
##features selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif, f_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile
#from mlxtend.feature_selection import SequentialFeatureSelector as SFS
##modeling the algorithms
from keras.utils import np_utils
#from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Embedding,Flatten
from keras.optimizers import RMSprop, SGD
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from keras import regularizers
from keras.layers import LSTM, SimpleRNN, GRU, Dense,RNN 
from keras.layers import Dropout
#from keras.model import model_from_json
from keras import callbacks
#evaluation metrics the algorithms
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
import scikitplot as skplt
from matplotlib import pyplot
#from keras.utils import plot_model               # sould be find a solution
from keras.models import model_from_json
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score,confusion_matrix,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
#from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")
#from keras.utils import plot_model # sould be find a solution next two line is a solution
import tensorflow as tf
from keras.utils.vis_utils import plot_model

print(tf.__version__)
print(np.__version__)
print(pd.__version__)
print(sklearn.__version__)
print(sk.__version__)
print(sn.__version__)
#print(plt.__version__)

#Load the Dataset
# attach the column names to the dataset
#dataset field name 
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

# KDDTrain+.csv & KDDTest+.csv are the datafiles without the last 
#column about the difficulty score these have already been removed.
# load train and test dataset 
df = pd.read_csv("NSL_KDD_Train.csv", header=None, names = col_names) #~/Desktop/dataset/
testdf = pd.read_csv("NSL_KDD_Test.csv", header=None, names = col_names)
# shape, this gives the dimensions of the dataset
print('Dimensions of the Training set:',df.shape)
print('Dimensions of the Test set:',testdf.shape)
print("Total number: ", len(df)+len(testdf))

df.dropna()

testdf.dropna()

df.isnull().values.any()

testdf.isnull().values.any()

df.isnull().sum().sum()

testdf.isnull().sum().sum()

df.dropna(axis='columns')

testdf.dropna(axis='columns')

df.head()

testdf.head()

df.isnull()

testdf.isnull()

#Statistical Summary
df.describe() 

#data viulizations
print(df.info())
print(testdf.info())

#Label Distribution of Training and Test set
print('Label distribution Training set:')
print(df['label'].value_counts())
print()
print('Label distribution Test set:')
print(testdf['label'].value_counts())

#Identify categorical features
# colums that are categorical and not binary yet: protocol_type (column 2), service (column 3), flag (column 4).
# explore categorical features
print('Training set:')
for col_name in df.columns:
    if df[col_name].dtypes == 'object' :
        unique_cat = len(df[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

#see how distributed the feature service is, it is evenly distributed and therefore we need to make dummies for all.
print()
print('Distribution of categories in service:')
print(df['service'].value_counts().sort_values(ascending=False).head())

# Test set
print('Test set:')
for col_name in testdf.columns:
    if testdf[col_name].dtypes == 'object' :
        unique_cat = len(testdf[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))
        
#LabelEncoder
#Insert categorical features into a 2D numpy array
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
categorical_columns=['protocol_type', 'service', 'flag']
# insert code to get a list of categorical columns into a variable, categorical_columns
categorical_columns=['protocol_type', 'service', 'flag'] 
 # Get the categorical values into a 2D numpy array
df_categorical_values = df[categorical_columns]
testdf_categorical_values = testdf[categorical_columns]
df_categorical_values.head(5)

###Make column names for dummies for training set and testing set
## training set
# protocol type
unique_protocol=sorted(df.protocol_type.unique())
string1 = 'Protocol_type_'
unique_protocol2=[string1 + x for x in unique_protocol]
# service
unique_service=sorted(df.service.unique())
string2 = 'service_'
unique_service2=[string2 + x for x in unique_service]
# flag
unique_flag=sorted(df.flag.unique())
string3 = 'flag_'
#unique_flag2=[string3 + str(x) for x in unique_flag] # mine
unique_flag2=[string3 + x for x in unique_flag] #original
# put together
dumcols=unique_protocol2 + unique_service2 + unique_flag2
print(dumcols)

# for test set
unique_service_test=sorted(testdf.service.unique())
unique_service2_test=[string2 + x for x in unique_service_test]
testdfumcols=unique_protocol2 + unique_service2_test + unique_flag2
print(dumcols)

f_protocol = pd.crosstab(index=df["protocol_type"],columns="count")
f_protocol = f_protocol/len(df)
f_protocol = f_protocol[f_protocol["count"] > 0.01]

f_attacks = pd.crosstab(index=df["label"],columns="count")
f_attacks = f_attacks/len(df)
f_attacks = f_attacks[f_attacks["count"] > 0.01]

f_data = df[df['protocol_type'].isin(list(f_protocol.index))]
f_data = f_data[f_data['label'].isin(list(f_attacks.index))]

## Create a Two-Way Table
relationship_protocoal_attack = pd.crosstab(index=f_data["label"], 
                          columns=f_data["protocol_type"])
## Plot the Two-Way Table
relationship_protocoal_attack.plot(kind="bar", 
                 figsize=(8,8),
                 stacked=True);

## Create a cross tab dataframe
protocol_data = pd.crosstab(index = df["protocol_type"],columns="Protocol type")
frequency_table_protocol = (protocol_data/protocol_data.sum())

## Plot the dataframe
frequency_table_protocol.plot.bar();

#Transform categorical features into numbers using LabelEncoder
from sklearn.preprocessing import LabelEncoder
#train set
df_categorical_values_enc=df_categorical_values.apply(LabelEncoder().fit_transform)
print(df_categorical_values_enc.head(10))
# test set
testdf_categorical_values_enc=testdf_categorical_values.apply(LabelEncoder().fit_transform)
print(testdf_categorical_values_enc.head())

# test set
testdf_categorical_values_enc=testdf_categorical_values.apply(LabelEncoder().fit_transform)
print(testdf_categorical_values_enc.head())
# i add the ff 3 line of code
#le = preprocessing.labelEncoder()

#One-Hot-Encoding
from sklearn.preprocessing import OneHotEncoder
# for train dataset
df = df.loc[df.index.drop_duplicates()]
testdf = testdf.loc[testdf.index.drop_duplicates()]
enc = OneHotEncoder()
df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(),columns=dumcols)
df_cat_data.head()
# test dataset
testdf_categorical_values_encenc = enc.fit_transform(testdf_categorical_values_enc)
testdf_cat_data = pd.DataFrame(testdf_categorical_values_encenc.toarray(),columns=testdfumcols)
df_cat_data.head()
testdf_cat_data.head()

# test set
testdf_categorical_values_encenc = enc.fit_transform(testdf_categorical_values_enc)
testdf_cat_data = pd.DataFrame(testdf_categorical_values_encenc.toarray(),columns=testdfumcols)

#testd[cat]=le.fit_transform(testd[cat].astype(float)) # i add it to fix list and float issue
#testd[cat]=testd[cat].astype('category') # i add it 
testdf_cat_data.head()

testdf_cat_data.head()

#Add 6 missing categories from train set to test set
trainservice=df['service'].tolist()
testservice= testdf['service'].tolist()
difference=list(set(trainservice) - set(testservice))
string = 'service_'
difference=[string + x for x in difference]
difference

for col in difference:
    testdf_cat_data[col] = 0

testdf_cat_data.shape

for col in difference:
    df_cat_data[col] = 0

df_cat_data.shape

#Join encoded categorical dataframe with the non-categorical dataframe
#obtaine 123 features previous 42 feature + current encoded 84 features - 3features (flag, protocol,services)
newdf=df.join(df_cat_data)
newdf.drop('flag', axis=1, inplace=True)
newdf.drop('protocol_type', axis=1, inplace=True)
newdf.drop('service', axis=1, inplace=True)
# test data
newtestdf=testdf.join(testdf_cat_data)
newtestdf.drop('flag', axis=1, inplace=True)
newtestdf.drop('protocol_type', axis=1, inplace=True)
newtestdf.drop('service', axis=1, inplace=True)
print(newdf.shape)
print(newtestdf.shape)

newdf.dropna(axis=1, inplace=True)
newtestdf.dropna(axis=1, inplace=True)

#split dataset in to 4 dataset for every attack category (Classification of attack) 
#rename every attack label: 0= normal,1=DOS,2=prope,3=R2L 4=U2R
#replace lable columns new lable columns 
# make new datasets
# take label column
labeldf=newdf['label']
labeltestdf=newtestdf['label']
# change the label column
newlabeldf=labeldf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
newlabeltestdf=labeltestdf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})

# Now put the new label column back
newdf['label'] = newlabeldf
newtestdf['label'] = newlabeltestdf
print(newdf['label'].head())
print(newtestdf['label'].head())

print(newdf.shape)
print(newtestdf.shape)

#######################3
#Convert "label" into normal=0 and attack=1 for KDDTrain+
newdf['class']=newdf['label'].apply(lambda x: 1 if x>=1 else 0)
newdf.drop(['label'], axis=1)

#Convert "label" into normal=0 and attack=1 for KDDTest+
newtestdf['class']=newtestdf['label'].apply(lambda x: 1 if x>=1 else 0)
newtestdf.drop(['label'],1)

newdf.groupby('class').count()

import matplotlib.pyplot as plt 
import seaborn as sns
sns.countplot(x="class", data=newdf, palette="Accent")
plt.title('Class Distributions in KDDTrain+ \n 0: Normal || 1: Attack', fontsize=14)
#plt.show()

newtestdf.groupby('class').count()

sns.countplot(x="class", data=newtestdf, palette="Accent")
plt.title('Class Distributions in KDDTest+ \n 0: Normal || 1: Attack', fontsize=14)
#plt.show()

# step1: apply the logarithmic scaling method for 
#scaling to obtain the ranges of `duration[0,4.77]',
#`src_bytes[0,9.11]' and `dst_bytes[0,9.11]
newdf['log2_value1'] = np.log2(newdf['duration'])
newdf['log2_value2'] = np.log2(newdf['src_bytes'])
newdf['log2_value3'] = np.log2(newdf['dst_bytes'])
newdf=newdf.drop(['log2_value3','log2_value2','log2_value1'], axis=1)

# testing set
newtestdf['log2_value1'] = np.log2(newtestdf['duration'])
newtestdf['log2_value2'] = np.log2(newtestdf['src_bytes'])
newtestdf['log2_value3'] = np.log2(newtestdf['dst_bytes'])
newtestdf=newtestdf.drop(['log2_value3','log2_value2','log2_value1'], axis=1)

print(newdf)
print(newtestdf)

x=newdf.drop('class',1) #X-train
y=newdf["class"] #y-Train
xtest=newtestdf.drop("class",1) #X-test
ytest=newtestdf['class'] # y-test
xtest

# Step 2: the value of every feature is mapped to the [0,1] range linearly
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
# Training Set
scale = MinMaxScaler()
scale= preprocessing.StandardScaler().fit(x)
x=scale.transform(x) 
scaletest= preprocessing.StandardScaler().fit(xtest)
xtest=scaletest.transform(xtest)

from tensorflow import keras
import numpy as np
import datetime
import time
x=pd.DataFrame(x)
x = x.values
sample = x.shape[0]
features = x.shape[1]
#Train: convert 2D to 3D for input RNN
x_train = np.reshape(x,(sample, 1, features)) #shape  = (125973, 18, 1)
#Test: convert 2D to 3D for input RNN
x_test=pd.DataFrame(xtest)
x_test = x_test.values
x_test = np.reshape(x_test,(x_test.shape[0], 1, x_test.shape[1]))

x_train.shape
x_test.shape

model3 = Sequential()
model3.add(LSTM(32, return_sequences=True, input_shape=(1,123))) #hidden1
model3.add(Dropout(0.6))
model3.add(LSTM(32, return_sequences=True)) #hidden2
model3.add(Dropout(0.6))
model3.add(LSTM(32, return_sequences=True))  # hidden3
model3.add(Dropout(0.6))
model3.add(LSTM(32, return_sequences=False)) #hidden4
model3.add(Dropout(0.6))
model3.add(Dense(1, activation = 'sigmoid')) #out put layers
model3.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics=['accuracy'])#binary_crossentropy
model3.summary()

model3.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.0001,patience=5) ## early stoppoing
#history3 = model3.fit(x_train, y, batch_size=32, epochs=100, validation_data=(x_test, ytest),callbacks=[es])
history3 = model3.fit(x_train, y, validation_data = (x_test, ytest), epochs=100, batch_size = 5000)

#print(model.get_config())
## fit the model...
#print("Train and Test attack using LSTM.")
#history3 = model2.fit(x_train, y, batch_size=32, epochs=100, verbose=1, validation_data=(x_test, ytest))

y_pred_lstm = model3.predict_classes(x_test)
y_probs_lstm=model3.predict_proba(x_test)
######Plot confusion matrix
skplt.metrics.plot_confusion_matrix(ytest, y_pred_lstm)
plt.title("LSTM-Confusion Matrix")
plt.show()

loss,accuracy = model3.evaluate(x_test, ytest)
print("\n Loss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

accuracy = accuracy_score(ytest, y_pred_lstm)
print("accuracy:",accuracy)
f1score=f1_score(ytest, y_pred_lstm)
print("f1-acore:",f1score)
cm=confusion_matrix(ytest, y_pred_lstm)
print("confusion matrix:\n",cm)
pr=precision_score(ytest,y_pred_lstm)
print("Precision:",pr)
rs=recall_score(ytest,y_pred_lstm)
print("Recall_score:",rs)

plt.figure(0) 
plt.plot(history3.history['accuracy'])
plt.plot(history3.history['val_accuracy'])
plt.rcParams['figure.figsize'] = (10, 5)
plt.xlabel("Numb of Epochs") 
plt.ylabel("Accuracy") 
plt.title(" LSTM model to detect Attack)")
plt.legend(['x_train','x_test'], loc='best')
plt.xlim([1,100])
plt.ylim([0,1.1])
plt.show()

plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
plt.rcParams['figure.figsize'] = (10, 5)
plt.title('model loss on the MLP model')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.title(" LSTM model to detect Attack)")
plt.legend(['x_train','x_test'], loc='best')
plt.xlim([1,100])
plt.ylim([-0.1,1])
plt.show()

