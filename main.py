import pandas as pd

from keras_predictions import KerasPredictions

# df = pd.read_excel('assets/excel/Budget for the North Channel edited.xlsx', sheet_name='Sheet2')
# df = pd.read_csv('assets/excel/Budget for the North Channel edited 2.csv')
# df = pd.read_csv('jena_climate_2009_2016.csv')
df = pd.read_excel('assets/excel/بيانات سد الوحدة.xlsx', sheet_name='Sheet2')
# df = pd.read_csv('assets/excel/2بيانات سد الوحدة.csv')

# df.index = pd.to_datetime(df['Date'], format='%d/%m/%Y')
# df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
df.index = pd.to_datetime(df['DAM_DAILY_DATE'])

df_index = df.index

# Selecting the column for prediction
# df = df['Tiberias to the Canal'].values.reshape(-1, 1)
# df = df['The lenticular'].values.reshape(-1, 1)
# column = df['T (degC)'].values.reshape(-1, 1)
df = df['TOTAL_INFLOW'].values.reshape(-1, 1)
# df = df['TOTAL_OUTFLOW'].values.reshape(-1, 1)
# df = df['DAM_DAILY_LEVEL'].values.reshape(-1, 1)

predictions = KerasPredictions(df=df, df_index=df_index)

predictions.train_model(model_name='dam_inflow')
# predictions.train_dams(model_name='dams')
# predictions.train_with_best_batch_size()
# predictions.get_predictions(model_name='tiberias')
# predictions.get_dams_predictions(model_name='dam_inflow',future_steps=7)
# predictions.get_channel_predictions(model_name='model_8')
