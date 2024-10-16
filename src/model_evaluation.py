import numpy as np 
import pandas as pd

import pickle
import json
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score , roc_auc_score

clf=pickle.load(open('models/model.pkl','rb'))

#fetch the data from raw/features
test_data=pd.read_csv('./data/features/test_bow.csv')

X_test=test_data.iloc[:,0:-1]
y_test=test_data.iloc[:,-1]

# Make predictions
y_pred = clf.predict(X_test)


# Make predictions
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred) 
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

metrics_dict={
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall,
    'auc':auc
}

data_path=os.path.join("metrics")
if not os.path.exists(data_path):
    os.makedirs(data_path)


with open('./metrics/metrics.json','w') as file:
    json.dump(metrics_dict,file,indent=4)