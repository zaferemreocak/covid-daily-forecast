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
  * C = 1
  ###### Model fitness:
  * As trying different models, a fitness parameter was required. So, accuracy variable is used.
  * accuracy = abs(prediction_values - real_values).mean()
  * The bigger value of accuracy, the less fit the model is.
  ###### Model saving:
  * depends on user action, if user decides to save model upon comparing prediction and real test data gap.
  * scalers of training and test is saved for future inverse_transform.
  
  #### predict.py
	gets 3 parameter(total active case, total confirmed case, total recored case) and predict # of death for the day.
  - Sample Prediction Scenario:
  ###### >> Enter # of active case for today: 21451
  ###### >> Enter # of confirmed case for today: 173036
  ###### >> Enter # of recovered case for today: 146839
  ###### >> 18.712
