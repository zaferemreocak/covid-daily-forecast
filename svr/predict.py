# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 01:47:08 2020

@author: Emre
"""

from joblib import load

def predict():
    
    #   reload SVM
    _svr = load("covid19.svr")
    #   reload the standart scaler
    scaler_x = load("scaler_x.bin")
    scaler_y = load("scaler_y.bin")
    #   get input from user
    active = input("Enter # of total active case: ")
    confirmed = input("Enter # of total case: ")
    recovered = input("Enter # of total recovered: ")
    #   make prediction and display
    print(scaler_y.inverse_transform(_svr.predict(scaler_x.transform([[active, confirmed, recovered]]))))
    
predict()