import numpy as np
import pandas as pd
import os
pd.set_option('future.no_silent_downcasting', True)

from sklearn.model_selection import train_test_split

df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

df.drop(columns=['tweet_id'], inplace=True)

final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()  # Use .copy() to avoid 

final_df['sentiment'] = final_df['sentiment'].replace({'happiness': 1, 'sadness': 0})

train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)

os.makedirs('data/raw',exist_ok=True)

train_data.to_csv('data/raw/train.csv',index=False)
test_data.to_csv('data/raw/test.csv',index=False)