# Covid19

A Neural Net predicts how many people will be death for the day after looking at some parameters which are stated below.

### train.py
	fetch, preprocess and train the model before making predictions.
##### /for collecting dataset	:	"https://api.covid19api.com/total/dayone/country/turkey"
##### /for preprocessing : from sklearn.preprocessing import StandardScaler
##### /for training : import tensorflow
  To create, compile and fit the model, Keras of TensorFlow is utilized.
  ###### Dataset Attributes:
  * number of features is 3 (total active case, total confirmed case, total recored case)
  * label : # of death people for each day
  ###### Training and Test dataset:
  * training data (the first 99.5% of forecast is training data)
  * test data (the last 0.05% of forecast is test data, shuffle is disabled)
  ###### Model architecture is:
  * 1 input layer (# of unit=8, activation='relu')
  * 2 hidden layer (# of unit=8, activation='relu')
  * 1 output layer (# of unit=1)
  ###### Model compilation:
  * optimizer = 'adam'
  * loss = 'mean_squared_error' (Since this is a regression problem, the model is need a mean squared error for loss function.)
  ###### Model fitting:
  * batch_size = 2
  * epoch = 10
  ###### Model saving:
  * depends on user action, if user decides to save model upon comparing prediction and real test data gap.
  * scalers of training and test is saved for future inverse_transform.

### predict.py
	gets 3 parameter(total active case, total confirmed case, total recored case) and predict # of death for the day.
  - [Prediction / Actual] Numbers of previously trained model for the last 5 days of covid pandemic:
  ###### 16.29431 / 21
  ###### 20.560995 / 23
  ###### 30.694738 / 19
  ###### 29.182936 / 18
  ###### 17.417822 / 17
  - Sample Prediction Scenario:
  ###### >> Enter # of active case for today: 21451
  ###### >> Enter # of confirmed case for today: 173036
  ###### >> Enter # of recovered case for today: 146839
  ###### >> 17.006247
