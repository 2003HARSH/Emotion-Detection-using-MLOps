import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os

#downloading raw data
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

#applying some basic changes
df.drop(columns=['tweet_id'],inplace=True)

final_df = df[df['sentiment'].isin(['happiness','sadness'])]

final_df['sentiment'].replace({'happiness':1, 'sadness':0},inplace=True)

#train_test_split
train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)


#storing data in data/raw
data_path=os.path.join("data","raw")

os.makedirs(data_path)

train_data.to_csv(os.path.join(data_path,"train.csv"))
test_data.to_csv(os.path.join(data_path,"test.csv"))