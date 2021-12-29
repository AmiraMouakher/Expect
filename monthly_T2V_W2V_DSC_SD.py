
# coding: utf-8

# In[73]:

import pandas as pd
import numpy
import matplotlib.pyplot as plt
# Spliting data to train and test:
from sklearn.model_selection import train_test_split
#  Evaluation metrics:
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import gc
import math
from sklearn.preprocessing import MinMaxScaler
#  Visualization packages:
import matplotlib.pyplot as plt
# Model training packages
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Layer
from keras.layers import Dense, Input, LSTM, Bidirectional, PReLU, LeakyReLU, Dropout,ZeroPadding1D
import keras.backend as K
from sklearn.impute import KNNImputer
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import * 
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
import keras


# In[2]:


# In[2]:
c = pd.read_csv('Data/data/TrainDCS-SK.csv')
info = pd.read_csv('Data/data/addinfoSamiDistance.csv')


# In[4]:


temp_df = []
for row in info.itertuples(index=False):
    temp_df.extend([list(row)]*17520)
new_info = pd.DataFrame(temp_df, columns=info.columns)
new_info['dwelling_type'] = new_info['dwelling_type'].factorize()[0]

# In[5]:



# # In[7]:

c['meter_id'] = new_info['meter_id']
c['dwelling_type'] = new_info['dwelling_type']
c['num_bedrooms'] = new_info['num_bedrooms']
# c
# # In[8]:
c = c.drop(columns=['Unnamed: 0'])
# # In[9]:
# Creating a dictionary that contains each road segment data speratly:
datas = {}
for i, g in c.groupby('meter_id'):
    datas.update({'meter_id_' + i : g})

# # time2Vec:

# In[ ]:


# from kerashypetune import KerasGridSearch


### DEFINE T2V LAYER ###

class T2V(Layer):
    
    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(T2V, self).__init__(**kwargs)
        
    def build(self, input_shape):

        self.W = self.add_weight(name='W',
                                shape=(input_shape[-1], self.output_dim),
                                initializer='uniform',
                                trainable=True)

        self.P = self.add_weight(name='P',
                                shape=(input_shape[1], self.output_dim),
                                initializer='uniform',
                                trainable=True)

        self.w = self.add_weight(name='w',
                                shape=(input_shape[1], 1),
                                initializer='uniform',
                                trainable=True)

        self.p = self.add_weight(name='p',
                                shape=(input_shape[1], 1),
                                initializer='uniform',
                                trainable=True)

        super(T2V, self).build(input_shape)
        
    def call(self, x):
        
        original = self.w * x + self.p
        sin_trans = K.sin(K.dot(x, self.W) + self.P)
        
        return K.concatenate([sin_trans, original], -1)
    
### DEFINE T2V LAYER ###

class W2V(Layer):
    
    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(W2V, self).__init__(**kwargs)
        
    def build(self, input_shape):

        self.W = self.add_weight(name='W',
                                shape=(input_shape[-1], self.output_dim),
                                initializer='uniform',
                                trainable=True)

        self.P = self.add_weight(name='P',
                                shape=(input_shape[1], self.output_dim),
                                initializer='uniform',
                                trainable=True)

        self.w = self.add_weight(name='w',
                                shape=(input_shape[1], 1),
                                initializer='uniform',
                                trainable=True)

        self.p = self.add_weight(name='p',
                                shape=(input_shape[1], 1),
                                initializer='uniform',
                                trainable=True)

        super(W2V, self).build(input_shape)
        
    def call(self, x):
        
        original = self.w * x + self.p
        sin_trans = K.sin(K.dot(x, self.W) + self.P)
        
        return K.concatenate([sin_trans, original], -1)
    
    

### DEFINE MODEL STRUCTURES ###

def T2V_W2V_NN(f, dim):
    Finp = Input(shape=(f,dim), name='Finp')
    Tinp = Input(shape=(1,dim),name='Tinp')
    Winp = Input(shape=(1,dim),name='Winp')
    t = T2V(dim)(Tinp)
    w = W2V(dim)(Winp)
    concat_1 = keras.layers.Concatenate()([t, w])
    concat_1 = ZeroPadding1D(padding = (0,15))(concat_1)
    concat = keras.layers.Concatenate()([Finp, concat_1])
    x = LSTM(200, return_sequences=True)(concat)
    x = LSTM(100)(x)
#     x = keras.layers.Concatenate(axis=1,return_sequences=False)([Finp, x])
#     x = Dense(48)(x)
    x = Dense(1)(x)
    m = Model(inputs = [Tinp, Winp, Finp], outputs=x)
    m.compile(loss='mse', optimizer='adam')
    return m   


# In[92]:


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back,look_after):
    dataF, dataX, dataW, dataY = [], [], [], []
    i = 0
    while i < len(dataset)-look_back :
        j = i + look_back
        cols = list(dataset.columns)
        cols.remove('meter_id')
        cols.remove('consommation')
        a = dataset[i:j][cols]
        dataF.append(a)
        a = dataset[i:j]['consommation']
        dataX.append(a)
        b = dataset[j:(j+look_after)]['consommation']
        dataY.append(np.sum(b.values))
        dataW.append(dataset[i:j]['weather_avg'])
        i = i + look_back
        
    return numpy.array(dataX), numpy.array(dataF),numpy.array(dataW), numpy.array(dataY)




# fix random seed for reproducibility
numpy.random.seed(7)
test_errors_lstm = pd.DataFrame(columns=['meter_Id','MAE','MSE','MAPE','RAE'],index=range(len(datas)))
prediction_results = pd.DataFrame(columns=['meter_Id','Actual_monthly','Predicted_monthly'],index=range(len(datas)*90))

import numpy as np
def RAE(actual, predicted):
    numerator = np.sum(np.abs(predicted - actual))
    denominator = np.sum(np.abs(np.mean(actual) - actual))
    return numerator / denominator


i = 0
for m in datas:
    # load the dataset
    dataset = datas[m]
#     dataset = pd.DataFrame(dataset.values.reshape(-1,48,int(dataset.shape[0]/48)).mean(1)).values.reshape(-1, 1).astype(float)
    
#     # normalize the dataset
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     dataset = scaler.fit_transform(dataset)

    look_back = 48 # Choose the number of steps that you want the model to look to predict the next step.
    look_after = 1440
    train, test = pd.DataFrame(columns=dataset.columns), pd.DataFrame(columns=dataset.columns)
    for r, g in dataset.groupby("season"):
        # split into train and test sets
        train_size = int(len(g) * 0.75)
        train_size = train_size - train_size%look_back
        test_size = len(g) - train_size
        test_size = test_size - test_size%look_back
        train = train.append(g[0:train_size])
        test = test.append(g[train_size:train_size+test_size])
    
    
        
        

    # reshape:

    trainX, trainF, trainW, trainY = create_dataset(train, look_back,look_after)
    testX,testF, testW, testY = create_dataset(test, look_back,look_after)
    
#     # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1,trainX.shape[1])).astype('float32')
    trainW = numpy.reshape(trainW, (trainW.shape[0], 1,trainW.shape[1])).astype('float32')
    trainF = numpy.reshape(trainF, (trainF.shape[0], 16,trainF.shape[1])).astype('float32')
    trainY = numpy.reshape(trainY, (trainY.shape[0], 1)).astype('float32')
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1])).astype('float32')
    testW = numpy.reshape(testW, (testW.shape[0], 1, testW.shape[1])).astype('float32')
    testF = numpy.reshape(testF, (testF.shape[0], 16, testF.shape[1])).astype('float32')
    testY = numpy.reshape(testY, (testY.shape[0], 1)).astype('float32')
    
    
 



    ### FIT T2V + LSTM ###

    print("---------------------------T2V + LSTM---------------------------")
    model = T2V_W2V_NN(f=trainF.shape[1], dim=look_back)
    print(model.summary())
    model.fit(x = [trainX,trainW, trainF], y = trainY, validation_split=0.1, epochs=200,verbose=1)

    pred_t2v = model.predict([testX,testW, testF]).ravel()
    
#     pred_t2v = scaler.inverse_transform(pred_t2v.reshape(-1, 1))
#     testY = scaler.inverse_transform(testY.ravel().reshape(-1, 1))
    ### VISUALIZE TEST PREDICTIONS ###

   
    # calculate Errors:
    print( 'Evaluation Results: ')
    print( '-------------------')
    print()

    rmse = math.sqrt(mean_squared_error(testY, pred_t2v))
    print('RMSE: ', rmse)
    print()

    mae = mean_absolute_error(testY, pred_t2v)

    print('MAE: %f' % mae)
    print()

    mse = mean_squared_error(testY, pred_t2v)

    print('MSE: %f' % mse)
    print()

    mape = numpy.mean(numpy.abs((testY - pred_t2v) / testY.ravel())) * 100
    print('MAPE: %f' % mape) 
    print()
    
    rae = RAE(testY.ravel() , pred_t2v.ravel())
    print('RAE: %f' % rae) 
    print()
    
    test_errors_lstm.iloc[i,0]= m
    test_errors_lstm.iloc[i,1]= mae
    test_errors_lstm.iloc[i,2]= mse
    test_errors_lstm.iloc[i,3]= mape
    test_errors_lstm.iloc[i,4]= rae
    
    for a in range(len(testY)):
        prediction_results.iloc[a,0]= m
        prediction_results.iloc[a,1]= testY[a][0]
        prediction_results.iloc[a,2]= pred_t2v[a]
        
    test_errors_lstm.to_csv('Errors_monthly_T2V_W2V_DSC_SK.csv',index=False)
    prediction_results.to_csv('PredVsActual_monthly_T2V_W2V_DSC_SK.csv',index=False)
    
    i = i + 1
    del model
    keras.backend.clear_session()
    gc.collect()   


# In[ ]:


print( '\n')
print( 'MAX MAE = ', max(test_errors_lstm['MAE']))
print( 'MAX MSE = ',max(test_errors_lstm['MSE']))
print( 'MAX MAPE = ',max(test_errors_lstm['MAPE']))
print( 'MAX RAE = ',max(test_errors_lstm['RAE']))
print( '\n')
print( 'MIN MAE = ', min(test_errors_lstm['MAE']))
print( 'MIN MSE = ',min(test_errors_lstm['MSE']))
print( 'MIN MAPE = ',min(test_errors_lstm['MAPE']))
print( 'MIN RAE = ',min(test_errors_lstm['RAE']))
print( '\n')
import numpy as np
print( 'MEAN MAE = ', np.mean(test_errors_lstm['MAE']))
print( 'MEAN MSE = ',np.mean(test_errors_lstm['MSE']))
print( 'MEAN MAPE = ',np.mean(test_errors_lstm['MAPE']))
print( 'MEAN RAE = ',np.mean(test_errors_lstm['RAE']))
print( '\n')