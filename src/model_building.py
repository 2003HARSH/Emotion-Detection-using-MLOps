import numpy as np
import pandas as pd
import pickle
import os
import yaml

from sklearn.ensemble import GradientBoostingClassifier

#fetch the data from raw/features
train_data=pd.read_csv('./data/features/train_bow.csv')

# Loading parameters from params.yaml
n_estimators=yaml.safe_load(open('./params.yaml','r'))['model_building']['n_estimators']
learning_rate=yaml.safe_load(open('./params.yaml','r'))['model_building']['learning_rate']

X_train=train_data.iloc[:,0:-1].values
y_train=train_data.iloc[:,-1].values

# Define and train the GBC model
clf = GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=learning_rate)
clf.fit(X_train,y_train)

#pickle the model
data_path=os.path.join("models")
if not os.path.exists(data_path):
    os.makedirs(data_path)
    
pickle.dump(clf,open('./models/model.pkl','wb'))