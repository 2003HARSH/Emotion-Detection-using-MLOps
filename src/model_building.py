import numpy as np
import pandas as pd
import pickle
import os

from sklearn.ensemble import GradientBoostingClassifier

#fetch the data from raw/features
train_data=pd.read_csv('./data/features/train_bow.csv')

X_train=train_data.iloc[:,0:-1].values
y_train=train_data.iloc[:,-1].values

# Define and train the GBC model
clf = GradientBoostingClassifier(n_estimators=50)
clf.fit(X_train,y_train)

#pickle the model
data_path=os.path.join("models")
os.makedirs(data_path)

pickle.dump(clf,open('./models/model.pkl','wb'))