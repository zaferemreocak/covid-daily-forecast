# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 22:25:18 2020

@author: Emre
"""

import requests
import numpy as np
import json
from sklearn.externals.joblib import dump

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
    #   split training and test data
    X = t_dataset[:, :-1].astype(float)
    y = t_dataset[:, -1].astype(float)
    
    y = y.reshape(-1, 1)
    #   scale the dataset
    from sklearn.preprocessing import StandardScaler
    scalerX = StandardScaler().fit(X)
    scalery = StandardScaler().fit(y)
    X = scalerX.transform(X)
    y = scalery.transform(y)
    
    return X, y, scalerX, scalery

def train():
    X, y, scalerX, scalery = preprocess()
    
    #   form a kernel and later, fit the training data
    from sklearn.svm import SVR
    svr_rbf = SVR(kernel='rbf', C=1)
    svr_rbf.fit(X, y.reshape(-1))
    #   calculate accuracy in percentage
    accuracy = scalery.inverse_transform(svr_rbf.predict(X)) - scalery.inverse_transform(y.ravel())
    accuracy = abs(accuracy).mean()
    #   print the predictions of the training data
    print("Accuracy: {}".format(accuracy))
    #   save the trained model
    check = input("Do you want to save the model? [YES/NO]: ")
    if(check.upper() == "YES"):
        dump(svr_rbf, "covid19.svr", compress=True)
        dump(scalerX, "scaler_x.bin", compress=True)
        dump(scalery, "scaler_y.bin", compress=True)
    
train()