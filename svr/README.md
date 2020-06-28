### SVR (Support Vector Regressor)
#### train.py
	Same concept as neural-network, the only difference lies on regression method.
##### /for training : from sklearn.svm import SVR
  To create and fit the model, SVR of sklearn is utilized. 
  ###### Dataset Attributes:
  * number of features is 3 (total active case, total confirmed case, total recored case)
  * label : # of death people for each day
  ###### Training and Test dataset:
  * all data is consumed by SVR model for fitting.
  ###### Model compilation:
  * kernel = 'rbf'
  * C = 8-repeat/5-fold cross-validation and available for user decision. init=.01 up-to 10^6
  ###### Model fitness:
  * fitness parameter is cross_val_score.mean() with cross_val_score.std() provided.
  ###### Model saving:
  * depends on user action, if user decides to save model upon fitness level of the current model.
  * scalers of training and test is saved for future inverse_transform.
  
  #### predict.py
	gets 3 parameter(total active case, total confirmed case, total recored case) and predict # of death for the day.
  - Sample Prediction Scenario:
  ###### >> Enter # of active case for today: 21451
  ###### >> Enter # of confirmed case for today: 173036
  ###### >> Enter # of recovered case for today: 146839
  ###### >> 18.712
