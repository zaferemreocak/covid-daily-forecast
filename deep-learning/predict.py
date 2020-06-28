# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:30:10 2020

@author: Emre
"""
import tensorflow as tf
from joblib import load

def predict():
    
    #   recreate the exact same model, including its weights and the optimizer
    ann = tf.keras.models.load_model("covid19.h5")
    
    #   reload the standart scaler
    scaler_x = load('scaler_x.bin')
    scaler_y = load('scaler_y.bin')
    
    #   get input from user
    active = input("Enter # of total active case: ")
    confirmed = input("Enter # of total case: ")
    recovered = input("Enter # of total recovered: ")
    
    #   make prediction and display
    print(scaler_y.inverse_transform(ann.predict(scaler_x.transform([[active, confirmed, recovered]]))))
    
predict()