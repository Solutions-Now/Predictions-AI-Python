import json

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from keras import Sequential
from keras.src.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.src.layers import LSTM, InputLayer, Dropout, Dense
from keras.src.losses import MeanSquaredError
from keras.src.metrics import RootMeanSquaredError
from keras.src.optimizers import Adam
from keras.src.saving import load_model

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


class KerasPredictions:

    def __init__(self, df, df_index, window_size=10):
        self.df_index = df_index
        self.scaler = MinMaxScaler()
        self.df = df
        self.window_size = window_size
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

    def df_to_X_y(self):
        self.df = self.scaler.fit_transform(self.df)
        X = []
        y = []
        for i in range(len(self.df) - self.window_size):
            X.append(self.df[i:i + self.window_size])
            y.append(self.df[i + self.window_size])
        return np.array(X), np.array(y)

    def train_model(self, model_name):
        X, y = self.df_to_X_y()
        # Split the data into training, validation, and test sets
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5,
                                                                            random_state=42)

        # Build the model
        model = Sequential()
        model.add(InputLayer(input_shape=(self.window_size, X.shape[2])))  # Input shape now includes the number of features
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(X.shape[2], activation='linear'))  # Output layer now has the same number of neurons as the number of features


        # Display the model summary
        model.summary()

        # Split the data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Compile the model
        cp = ModelCheckpoint(f'assets/models/{model_name}.keras', save_best_only=True)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0005), metrics=['mae'])

        # Train the model
        model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val),
                  callbacks=[cp, es, lr], batch_size=32)


    def get_predictions(self, model_name):

        # Load the best model
        model = load_model(f'assets/models/{model_name}.keras')
        X, y = self.df_to_X_y()
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Make predictions on training, validation, and test sets
        train_predictions = model.predict(X_train)
        val_predictions = model.predict(X_val)
        test_predictions = model.predict(X_test)

        # Inverse transform the predictions and the actual values to their original scale
        train_predictions = self.scaler.inverse_transform(train_predictions)
        y_train_original = self.scaler.inverse_transform(y_train)
        val_predictions = self.scaler.inverse_transform(val_predictions)
        y_val_original = self.scaler.inverse_transform(y_val)
        test_predictions = self.scaler.inverse_transform(test_predictions)
        y_test_original = self.scaler.inverse_transform(y_test)

        # Convert predictions to DataFrames for easy comparison
        train_result = pd.DataFrame(data=train_predictions, columns=['Train Predictions TOTAL_INFLOW', 'Train Predictions TOTAL_OUTFLOW', 'Train Predictions DAM_DAILY_LEVEL'])
        train_result['Actual TOTAL_INFLOW'] = y_train_original[:, 0]
        train_result['Actual TOTAL_OUTFLOW'] = y_train_original[:, 1]
        train_result['Actual DAM_DAILY_LEVEL'] = y_train_original[:, 2]

        val_result = pd.DataFrame(data=val_predictions, columns=['Val Predictions TOTAL_INFLOW', 'Val Predictions TOTAL_OUTFLOW', 'Val Predictions DAM_DAILY_LEVEL'])
        val_result['Actual TOTAL_INFLOW'] = y_val_original[:, 0]
        val_result['Actual TOTAL_OUTFLOW'] = y_val_original[:, 1]
        val_result['Actual DAM_DAILY_LEVEL'] = y_val_original[:, 2]

        test_result = pd.DataFrame(data=test_predictions, columns=['Test Predictions TOTAL_INFLOW', 'Test Predictions TOTAL_OUTFLOW', 'Test Predictions DAM_DAILY_LEVEL'])
        test_result['Actual TOTAL_INFLOW'] = y_test_original[:, 0]
        test_result['Actual TOTAL_OUTFLOW'] = y_test_original[:, 1]
        test_result['Actual DAM_DAILY_LEVEL'] = y_test_original[:, 2]


        # Assume your test set's last date is the starting point for future predictions
        last_date = self.df_index[-1]  # Last date in the dataset

        # Number of future steps to predict
        future_steps = 12  # Example: predict the next 12 days/months

        # Use the last window of data from your test set as the starting point
        last_window = X_test[-1]

        # Initialize a list to store future predictions
        future_predictions = []

        for _ in range(future_steps):
            # Predict the next set of values
            next_prediction = model.predict(last_window[np.newaxis, :, :]).flatten()
            future_predictions.append(next_prediction)

            # Update the window with the new prediction
            last_window = np.append(last_window[1:], [next_prediction], axis=0)

        # Inverse transform the predictions to original scale
        future_predictions_original_scale = self.scaler.inverse_transform(np.array(future_predictions))

        # Generate corresponding future dates (monthly/daily increments)
        future_dates = [last_date + relativedelta(days=i) for i in range(1, future_steps + 1)]

        # Create a DataFrame for future predictions
        future_result = pd.DataFrame(data=future_predictions_original_scale, columns=['TOTAL_INFLOW', 'TOTAL_OUTFLOW', 'DAM_DAILY_LEVEL'])
        future_result['Date'] = future_dates



        # Convert training predictions to a dictionary
        train_result_dict = {
            str(self.df_index[i]): {
                "TOTAL_INFLOW": float(train_predictions[i][0]),
                "TOTAL_OUTFLOW": float(train_predictions[i][1]),
                "DAM_DAILY_LEVEL": float(train_predictions[i][2])
            }
            for i in range(len(train_predictions))
        }

        # Convert validation predictions to a dictionary
        val_result_dict = {
            str(self.df_index[i]): {
                "TOTAL_INFLOW": float(val_predictions[i][0]),
                "TOTAL_OUTFLOW": float(val_predictions[i][1]),
                "DAM_DAILY_LEVEL": float(val_predictions[i][2])
            }
            for i in range(len(val_predictions))
        }

        # Convert test predictions to a dictionary
        test_result_dict = {
            str(self.df_index[i]): {
                "TOTAL_INFLOW": float(test_predictions[i][0]),
                "TOTAL_OUTFLOW": float(test_predictions[i][1]),
                "DAM_DAILY_LEVEL": float(test_predictions[i][2])
            }
            for i in range(len(test_predictions))
        }

        # Combine all into a single dictionary
        results_dict = {
            "Training Set Predictions": train_result_dict,
            "Validation Set Predictions": val_result_dict,
            "Test Set Predictions": test_result_dict,
        }

        # Convert the dictionary to a JSON string
        results_json = json.dumps(results_dict, indent=4)

        # Print the JSON formatted predictions
        print(results_json)


