import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os
import yaml

# Loading parameters from params.yaml
def load_params(path:str)->float:
    try:
        test_size=yaml.safe_load(open(path,'r'))['data_ingestion']['test_size']
        return test_size
    except FileNotFoundError:
        print(f'The file {path} was not found')
        raise
    except yaml.YAMLError:
        print(f'Error parsing YAML file {path}')
    except Exception as e:
        print(f'An error occurred: {e}')
        raise

#downloading raw data
def read_data(url:str)->pd.DataFrame:
    df = pd.read_csv(url)
    return df

#applying some basic changes
def process_data(df:pd.DataFrame)->pd.DataFrame:
    df.drop(columns=['tweet_id'],inplace=True)
    final_df = df[df['sentiment'].isin(['happiness','sadness'])]
    final_df['sentiment'].replace({'happiness':1, 'sadness':0},inplace=True)
    return final_df

#train_test_split
def train_test_split(final_df:pd.DataFrame,test_size:float)->tuple:
    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
    return train_data,test_data


#storing data in data/raw
def store_data(train_data:float,test_data:float)->None:
    data_path=os.path.join("data","raw")
    os.makedirs(data_path)
    train_data.to_csv(os.path.join(data_path,"train.csv"))
    test_data.to_csv(os.path.join(data_path,"test.csv"))

def main()->None:
    test_size=load_params('./params.yml')
    df=read_data('https://raw.githubusercontent.com/campus x-official/jupyter-masterclass/main/tweet_emotions.csv')
    final_df=process_data(df)
    train_data,test_data=train_test_split(final_df,test_size)
    store_data(train_data,test_data) 

if __name__--'__main__':
    main()