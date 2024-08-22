import pandas as pd

from keras_predictions import KerasPredictions
from steps_range import StepsRange

# df = pd.read_excel('assets/excel/بيانات سد الوحدة.xlsx', sheet_name='Sheet2')
df = pd.read_excel('assets/excel/طبريا يومي.xlsx', sheet_name='Sheet1')

# df.index = pd.to_datetime(df['DAM_DAILY_DATE'])
df.index = pd.to_datetime(df['Date'])
df_index = df.index
df = df.dropna()

# df = df[['TOTAL_INFLOW', 'TOTAL_OUTFLOW', 'DAM_DAILY_LEVEL']].values
# df = df[['TOTAL_INFLOW', 'TOTAL_OUTFLOW', 'DAM_DAILY_LEVEL', 'MAX_CAPACITY', 'TOTAL_VOLUME']].values
# df = df['TOTAL_INFLOW'].values.reshape(-1, 1)
df = df['Amount'].values.reshape(-1, 1)

# dam_daily_level_df = pd.read_excel('assets/excel/حجم بحيرة سد الوحده معدل.xlsx', sheet_name='المخزون الحديث')
# allowed_dam_daily_levels = dam_daily_level_df['values'].to_list()

predictions = KerasPredictions(df=df, df_index=df_index)
# predictions = KerasPredictions(df=df, df_index=df_index, allowed_dam_daily_levels=allowed_dam_daily_levels)

predictions.train_model(model_name='tiberias')
# predictions.get_predictions(model_name='tiberias', steps_range=StepsRange.DAYS,)
# predictions.train_model(model_name='dam_data', multiple_features=True)
# predictions.get_predictions(model_name='dam_data', multiple_features=True, steps_range=StepsRange.DAYS)
