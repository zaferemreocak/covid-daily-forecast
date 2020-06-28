# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 22:25:18 2020

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
    
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    #   scale the dataset
    from sklearn.preprocessing import StandardScaler
    scalerX = StandardScaler().fit(X_train)
    scalery = StandardScaler().fit(y_train)
    
    #   scale train data
    X_train = scalerX.transform(X_train).astype(float)
    y_train = scalery.transform(y_train).astype(float)
    
    #   scale test data
    X_test = scalerX.transform(X_test).astype(float)
    y_test = scalery.transform(y_test).astype(float)
    
    return X_train, X_test, y_train, y_test, scalerX, scalery

def train():
    X_train, X_test, y_train, y_test, scalerX, scalery = preprocess()
    
    #   form a kernel and later, fit the training data
    from sklearn.svm import SVR
    
    #   5-fold cross-validation
    from sklearn.model_selection import cross_val_score
    
    #   value of C variable
    var_c = 0.01
    for i in range (8):
        svr_rbf = SVR(kernel='rbf', C=var_c, gamma='auto')
        scores = cross_val_score(svr_rbf, X_train, y_train.reshape(-1), cv=5)
        print("[{3}]Accuracy: {0:.2f} (+/- {1:.2f}) with C={2}".format(scores.mean(), scores.std() * 2, var_c, i))
        var_c = var_c*10
    
    coeff = input("Enter a index number to select a C value: ")
    var_c = 0.01*(10**int(coeff))
    
    #   fit SVR model
    svr_rbf = SVR(kernel='rbf', C=var_c, gamma='auto')
    svr_rbf.fit(X_train, y_train.reshape(-1))
    
    #   predictions on the test data
    pred = scalery.inverse_transform(svr_rbf.predict(X_test)) - scalery.inverse_transform(y_test.reshape(-1))
    print("Accuracy on test data: {0:.2f} (+/- {1:.2f})".format(abs(pred.mean()), pred.std()))
    
    #   save the trained model
    check = input("Do you want to save the model? [y/n]: ")
    if(check.upper() == "Y"):
        dump(svr_rbf, "covid19.svr", compress=True)
        dump(scalerX, "scaler_x.bin", compress=True)
        dump(scalery, "scaler_y.bin", compress=True)
    
train()