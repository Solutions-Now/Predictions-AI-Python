import json

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from keras import Sequential
from keras.src.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.src.layers import LSTM, InputLayer, Dropout, Dense
from keras.src.losses import MeanSquaredError
from keras.src.metrics import RootMeanSquaredError
from keras.src.optimizers import Adam
from keras.src.saving import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class KerasPredictions:

    def __init__(self, df, df_index, window_size=10):
        self.df_index = df_index
        self.scaler = MinMaxScaler()
        self.df = df
        self.window_size = window_size

    def df_to_X_y(self):
        self.df = self.scaler.fit_transform(self.df)
        X = []
        y = []
        for i in range(len(self.df) - self.window_size):
            X.append(self.df[i:i + self.window_size])
            y.append(self.df[i + self.window_size])
        return np.array(X), np.array(y)

    def train_model(self, model_name,features_number = None):
        X, y = self.df_to_X_y()

        # Build the model
        model = Sequential()
        if features_number == None:
            model.add(InputLayer(input_shape=(self.window_size, 1)))
        else:
            model.add(InputLayer(input_shape=(self.window_size, X.shape[features_number - 1])))

        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64))
        model.add(Dense(32, activation='relu'))
        if features_number == None:
            model.add(Dense(1, activation='linear'))
        else:
            model.add(Dense(X.shape[features_number - 1], activation='linear'))

        # Display the model summary
        model.summary()

        # Split the data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Compile the model
        cp = ModelCheckpoint(f'assets/models/{model_name}.keras', save_best_only=True)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
        # Train the model
        model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val),
                  callbacks=[cp, es, lr], batch_size=32)

    def get_predictions(self, model_name,features_number = None):

        # Load the best model
        model = load_model(f'assets/models/{model_name}.keras')

        X, y = self.df_to_X_y()
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Assume your test set's last date is the starting point for future predictions
        last_date = self.df_index[-1]  # Last date in the dataset

        # Number of future steps to predict (30 days)
        future_steps = 30

        # Use the last window of data from your test set as the starting point
        last_window = X_test[-1]

        # Initialize a list to store future predictions
        future_predictions = []

        for _ in range(future_steps):
            # Predict the next set of values
            if features_number == None:
                next_prediction = model.predict(last_window[np.newaxis, :, :]).flatten()[0]
            else:
                next_prediction = model.predict(last_window[np.newaxis, :, :]).flatten()

            future_predictions.append(next_prediction)

            # Update the window with the new prediction
            if features_number == None:
                last_window = np.append(last_window[1:], [[next_prediction]], axis=0)
            else:
                last_window = np.append(last_window[1:], [next_prediction], axis=0)

        # Inverse transform the predictions to original scale
        if features_number == None:
            future_predictions_original_scale = self.scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
        else:
            future_predictions_original_scale = self.scaler.inverse_transform(np.array(future_predictions))

        # Generate corresponding future dates (daily increments)
        future_dates = [last_date + relativedelta(days=i) for i in range(1, future_steps + 1)]


        if features_number == None:
            # Convert predictions to float64 and create a dictionary of date-value pairs
            future_result = {str(date.date()): float(value) for date, value in
                             zip(future_dates, future_predictions_original_scale)}
            results_dict = future_result
        else:
            # Create a DataFrame for future predictions
            future_result = pd.DataFrame(data=future_predictions_original_scale,
                                         columns=['TOTAL_INFLOW', 'TOTAL_OUTFLOW', 'DAM_DAILY_LEVEL'])
            future_result['Date'] = future_dates
            # Convert future predictions to a dictionary
            future_result_dict = {
                str(future_dates[i]): {
                    "TOTAL_INFLOW": float(future_predictions_original_scale[i][0]),
                    "TOTAL_OUTFLOW": float(future_predictions_original_scale[i][1]),
                    "DAM_DAILY_LEVEL": float(future_predictions_original_scale[i][2])
                }
                for i in range(future_steps)
            }

            # Combine all into a single dictionary
            results_dict = {
                "Future Predictions (Next 30 Days)": future_result_dict
            }



        # Convert the dictionary to a JSON string
        results_json = json.dumps(results_dict, indent=4)

        # Print the JSON formatted predictions
        print(results_json)