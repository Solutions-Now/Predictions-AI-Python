import pandas as pd

from Predictions import Predictions

df = pd.read_excel('Budget for the North Channel edited.xlsx', sheet_name='Sheet2')
df.index = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df_index = df.index
# Selecting the column for prediction
# column = df['T (degC)'].values.reshape(-1, 1)
df = df['Tiberias to the Canal'].values.reshape(-1, 1)

predictions = Predictions(df=df, df_index=df_index)

predictions.train_model()
predictions.get_predictions()
