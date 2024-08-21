import pandas as pd

from keras_predictions import KerasPredictions

df = pd.read_excel('assets/excel/بيانات سد الوحدة.xlsx', sheet_name='Sheet2')

df.index = pd.to_datetime(df['DAM_DAILY_DATE'])

df_index = df.index

# Selecting the column for prediction

# Selecting the columns for prediction (e.g., TOTAL_INFLOW, TOTAL_OUTFLOW, DAM_DAILY_LEVEL)
df = df[['TOTAL_INFLOW', 'TOTAL_OUTFLOW', 'DAM_DAILY_LEVEL']].values

# df = df['TOTAL_INFLOW'].values.reshape(-1, 1)

predictions = KerasPredictions(df=df, df_index=df_index)

# predictions.train_model(model_name='dam_inflow')

predictions.get_predictions(model_name='dam_inflow')