import json

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from keras import Sequential
from keras.src.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.src.layers import LSTM, InputLayer, Dropout, Dense, LeakyReLU
from keras.src.losses import MeanSquaredError
from keras.src.metrics import RootMeanSquaredError
from keras.src.optimizers import Adam
from keras.src.saving import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from steps_range import StepsRange


class KerasPredictions:

    def __init__(self, df, df_index, allowed_dam_daily_levels = None, window_size=10):
        self.df_index = df_index
        self.scaler = MinMaxScaler()
        self.df = df
        self.window_size = window_size
        self.allowed_dam_daily_levels = allowed_dam_daily_levels

    def df_to_X_y(self):
        self.df = self.scaler.fit_transform(self.df)
        X = []
        y = []
        for i in range(len(self.df) - self.window_size):
            X.append(self.df[i:i + self.window_size])
            y.append(self.df[i + self.window_size])
        return np.array(X), np.array(y)

    def train_model(self, model_name, multiple_features=False):
        X, y = self.df_to_X_y()

        # Build the model
        model = Sequential()
        if not multiple_features:
            model.add(InputLayer(input_shape=(self.window_size, 1)))
        else:
            model.add(InputLayer(input_shape=(self.window_size, X.shape[2])))

        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64))
        model.add(Dense(32, activation='relu'))
        if not multiple_features:
            model.add(Dense(1, activation='linear'))
        else:
            model.add(Dense(X.shape[2], activation='linear'))

        # if not multiple_features:
        #     model.add(InputLayer(input_shape=(self.window_size, 1)))
        # else:
        #     model.add(InputLayer(input_shape=(self.window_size, X.shape[2])))
        #
        # model.add(LSTM(128, return_sequences=True))
        # model.add(Dropout(0.3))
        # model.add(LSTM(64, return_sequences=True))
        # model.add(Dropout(0.3))
        # model.add(LSTM(32))
        # model.add(Dense(64))
        # model.add(LeakyReLU(alpha=0.1))
        # if not multiple_features:
        #     model.add(Dense(1))
        # else:
        #     model.add(Dense(X.shape[2]))

        # Display the model summary
        model.summary()

        # Split the data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Compile the model
        cp = ModelCheckpoint(f'assets/models/{model_name}.keras', save_best_only=True)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
        lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
        # Train the model
        model.fit(X_train, y_train, epochs=150, validation_data=(X_val, y_val),
                  callbacks=[cp, es, lr], batch_size=16)

    def get_predictions(self, model_name, multiple_features=False, steps_range=StepsRange.DAYS, future_steps=30):

        # Load the best model
        model = load_model(f'assets/models/{model_name}.keras')

        X, y = self.df_to_X_y()
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Assume your test set's last date is the starting point for future predictions
        last_date = self.df_index[-1]  # Last date in the dataset

        # Number of future steps to predict (30 days)
        # future_steps = 30

        # Use the last window of data from your test set as the starting point
        last_window = X_test[-1]

        # Initialize a list to store future predictions
        future_predictions = []

        for _ in range(future_steps):
            # Predict the next set of values
            if not multiple_features:
                next_prediction = model.predict(last_window[np.newaxis, :, :]).flatten()[0]
            else:
                next_prediction = model.predict(last_window[np.newaxis, :, :]).flatten()

            future_predictions.append(next_prediction)

            # Update the window with the new prediction
            if not multiple_features:
                last_window = np.append(last_window[1:], [[next_prediction]], axis=0)
            else:
                last_window = np.append(last_window[1:], [next_prediction], axis=0)

        # Inverse transform the predictions to original scale
        if not multiple_features:
            future_predictions_original_scale = self.scaler.inverse_transform(
                np.array(future_predictions).reshape(-1, 1)).flatten()
        else:
            future_predictions_original_scale = self.scaler.inverse_transform(np.array(future_predictions))

        # Generate corresponding future dates (daily increments)
        future_dates = []
        for i in range(1, future_steps + 1):
            days = i if steps_range == StepsRange.DAYS else 0
            months = i if steps_range == StepsRange.MONTHS else 0
            years = i if steps_range == StepsRange.YEARS else 0
            future_date = last_date + relativedelta(days=days, months=months, years=years)
            future_dates.append(future_date)

        if not multiple_features:
            # Convert predictions to float64 and create a dictionary of date-value pairs
            future_result = {str(date.date()): float(value) for date, value in
                             zip(future_dates, future_predictions_original_scale)}
            results_dict = future_result
        else:
            # Create a DataFrame for future predictions
            future_result = pd.DataFrame(data=future_predictions_original_scale,
                                         columns=['TOTAL_INFLOW', 'TOTAL_OUTFLOW', 'DAM_DAILY_LEVEL', 'MAX_CAPACITY',
                                                  'TOTAL_VOLUME'])
            future_result['Date'] = future_dates
            # Convert future predictions to a dictionary
            future_result_dict = {}
            for i in range(future_steps):
                date_str = str(future_dates[i])
                total_inflow = float(future_predictions_original_scale[i][0])
                total_outflow = float(future_predictions_original_scale[i][1])
                dam_daily_level = self.get_closest_dam_daily_level(float(future_predictions_original_scale[i][2]))
                max_capacity = float(future_predictions_original_scale[i][3])
                total_volume = float(future_predictions_original_scale[i][4])
                storage_percentage = (total_volume / max_capacity) * 100
                future_result_dict[date_str] = {
                    "TOTAL_INFLOW": total_inflow,
                    "TOTAL_OUTFLOW": total_outflow,
                    "DAM_DAILY_LEVEL": dam_daily_level,
                    "MAX_CAPACITY": max_capacity,
                    "TOTAL_VOLUME": total_volume,
                    "STORAGE_PERCENTAGE": storage_percentage
                }

            # Combine all into a single dictionary
            results_dict = {
                "Future Predictions (Next 30 Days)": future_result_dict
            }

        # Convert the dictionary to a JSON string
        results_json = json.dumps(results_dict, indent=4)

        # Print the JSON formatted predictions
        print(results_json)

    def get_closest_dam_daily_level(self, prediction):
        # Round the prediction to one decimal place
        rounded_prediction = round(prediction, 1)
        if self.allowed_dam_daily_levels is None: return rounded_prediction
        allowed_values = self.allowed_dam_daily_levels
        # Find the closest value in allowed_values
        min_value = min(allowed_values, key=lambda x: abs(x - rounded_prediction))
        return round(min_value, 1)
