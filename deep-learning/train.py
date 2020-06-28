# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 20:57:10 2020

@author: Emre
"""

import requests
import numpy as np
import json
from joblib import dump

def fetch():
    
    #   URL end-point base
    url = "https://api.covid19api.com/total/dayone/country/turkey"
    
    #   send get request
    raw = requests.get(url)
    
    #   format json
    raw_json = raw.json()
    
    global dataset
        
    #   extract features from the dataset
    for i in range(len(raw_json)):
        #   parse json
        t = json.loads(json.dumps(raw_json[i]))
        active = t.get("Active")
        confirmed = t.get("Confirmed")
        death = t.get("Deaths")
        recovered = t.get("Recovered")
        
        #   initialization the very first day
        if i == 0 :
            dataset = np.array([[active, confirmed, recovered, death]])
        else:
            #   get day+1 and append the new row
            t_1 = json.loads(json.dumps(raw_json[i-1]))
            active = t.get("Active")
            confirmed = t.get("Confirmed")
            recovered = t.get("Recovered")
            death = t.get("Deaths") - t_1.get("Deaths")
            
            row = np.array([[active, confirmed, recovered, death]])
            dataset = np.vstack((dataset, row))
        
    return dataset
            
def preprocess():
    
    global t_dataset
    
    #   fetch and form the dataset
    t_dataset = fetch()
    
    #   due to API corruption, I need to make a hand-fix at day:105 and day:106
    # t_dataset[105] = [22398, 191657, 164234, 24]
    # t_dataset[106,3] = 21
    
    #   split training and test data
    X = t_dataset[:, :-1].astype(float)
    y = t_dataset[:, -1].astype(float)
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)
    
    #   scale the dataset
    from sklearn.preprocessing import StandardScaler
    scalerX = StandardScaler().fit(X_train)
    y_train = y_train.reshape(-1, 1)
    scalery = StandardScaler().fit(y_train)
    X_train = scalerX.transform(X_train)
    y_train = scalery.transform(y_train)
    X_test = scalerX.transform(X_test)
    y_test = y_test.reshape(-1, 1)
    y_test = scalery.transform(y_test)
    
    return X_train, X_test, y_train, y_test, scalerX, scalery
    
def train():
    
    X_train, X_test, y_train, y_test, scalerX, scalery = preprocess()
    
    import tensorflow as tf
    
    #   initializing the ANN
    ann = tf.keras.models.Sequential()
    
    #   adding the input layer and the first hidden layer
    ann.add(tf.keras.layers.Dense(units=8, activation='relu', input_shape=(3,)))
    ann.add(tf.keras.layers.Dense(units=8, activation='relu'))
    
    #   adding the output layer
    ann.add(tf.keras.layers.Dense(units=1))
    
    #   compiling the ANN
    ann.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    #   training the ANN on the training set
    ann.fit(X_train, y_train, batch_size = 2, epochs = 32)
    
    #   make predictions based on trained model
    y_pred = ann.predict(X_test)
    
    #   calculate margin of error
    pred = y_pred - y_test
    print("Margin of error for test data (scaled to range 0-1): {0:.5f}"
          .format(abs(pred.mean())))
    
    #   inverse-transformation
    y_pred = scalery.inverse_transform(y_pred)
    y_test = scalery.inverse_transform(y_test.reshape(-1))  

    #   save the trained model
    check = input("Do you want to save the model? [y/n]: ")
    if(check.upper() == "Y"):
        ann.save("covid19.h5")
        dump(scalerX, 'scaler_x.bin', compress=True)
        dump(scalery, 'scaler_y.bin', compress=True)
    
#    plt.plot(y_test, color = 'red', label = 'Real data')
#    plt.plot(y_pred, color = 'blue', label = 'Predicted data')
#    plt.title('Prediction')
#    plt.legend()
#    plt.show()
    
    
train()
