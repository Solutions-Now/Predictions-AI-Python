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

    # def train_model(self, model_name):
    #     X, y = self.df_to_X_y()
    #     # Split the data into training, validation, and test sets
    #     self.X_train, X_temp, self.y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    #     self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5,
    #                                                                         random_state=42)
    #
    #     # Build the model
    #     model1 = Sequential()
    #     model1.add(InputLayer(input_shape=(self.window_size, 1)))
    #     model1.add(LSTM(64, return_sequences=True))
    #     model1.add(LSTM(32))
    #     model1.add(Dense(16, activation='relu'))
    #     model1.add(Dense(1, activation='linear'))
    #
    #     # Display the model summary
    #     model1.summary()
    #
    #     # Compile the model
    #     cp = ModelCheckpoint(f'assets/models/{model_name}.keras', save_best_only=True)
    #     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    #     model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])
    #
    #     # Train the model
    #     model1.fit(self.X_train, self.y_train, epochs=50, validation_data=(self.X_val, self.y_val), callbacks=[cp, es],
    #                batch_size=16)

    def train_model(self, model_name):
        X, y = self.df_to_X_y()
        # Split the data into training, validation, and test sets
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5,
                                                                            random_state=42)

        # Build the model
        model1 = Sequential()
        model1.add(InputLayer(input_shape=(self.window_size, 1)))
        model1.add(LSTM(128, return_sequences=True))
        model1.add(Dropout(0.2))
        model1.add(LSTM(64))
        model1.add(Dense(32, activation='relu'))
        model1.add(Dense(1, activation='linear'))

        # Display the model summary
        model1.summary()

        # Compile the model
        cp = ModelCheckpoint(f'assets/models/{model_name}.keras', save_best_only=True)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
        model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0005), metrics=[RootMeanSquaredError()])

        # Train the model
        model1.fit(self.X_train, self.y_train, epochs=100, validation_data=(self.X_val, self.y_val),
                   callbacks=[cp, lr], batch_size=32)

    def train_with_best_batch_size(self):
        X, y = self.df_to_X_y()
        # Split the data into training, validation, and test sets
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5,
                                                                            random_state=42)
        batch_sizes = [1, 2, 4, 8, 10, 16, 30, 32, 64, 90, 128]
        best_batch_size = None
        best_val_loss = float('inf')

        for batch_size in batch_sizes:
            print(f"Training with batch size: {batch_size}")

            # Build and compile the model
            model1 = Sequential()
            model1.add(InputLayer(input_shape=(self.window_size, 1)))
            model1.add(LSTM(64, return_sequences=True))
            model1.add(LSTM(32))
            model1.add(Dense(16, activation='relu'))
            model1.add(Dense(1, activation='linear'))

            model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001),
                           metrics=[RootMeanSquaredError()])

            # Callbacks
            cp = ModelCheckpoint(f'assets/models/model_{batch_size}.keras', save_best_only=True)
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

            # Train the model
            history = model1.fit(self.X_train, self.y_train, epochs=50, validation_data=(self.X_val, self.y_val),
                                 callbacks=[cp, es], batch_size=batch_size)

            # Evaluate on validation set
            val_loss = min(history.history['val_loss'])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_batch_size = batch_size

        print(f"Best batch size: {best_batch_size} with validation loss: {best_val_loss}")

        return best_batch_size

    def train_dams(self, model_name):
        # Use historical data for features
        X = self.df[['TOTAL_INFLOW', 'TOTAL_OUTFLOW']].shift(1).dropna()
        y = self.df[['TOTAL_INFLOW', 'TOTAL_OUTFLOW']].iloc[1:]
        dates = self.df['DAM_DAILY_DATE'].iloc[1:]

        # Scaling data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        # # Split data
        # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

        # Split data
        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
            X_scaled, y_scaled, dates, test_size=0.2, random_state=42
        )

        # Build the model
        model = Sequential()
        model.add(InputLayer(input_shape=(X_train.shape[1],)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(2))  # 2 output units for TOTAL_INFLOW and TOTAL_OUTFLOW

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2)

        # Predict the next 30 days
        last_known_data = X_scaled[-1:]  # Get the last known data point
        predictions_dict = {}
        date = self.df.index[-1]  # Start from the last date in the dataset

        for i in range(30):
            # Make a prediction
            prediction_scaled = model.predict(last_known_data)
            prediction = scaler_y.inverse_transform(prediction_scaled)

            # Advance the date by one day
            date += pd.Timedelta(days=1)

            # Store the prediction in the dictionary
            predictions_dict[str(date.date())] = {
                "in_flow": float(prediction[0, 0]),
                "out_flow": float(prediction[0, 1])
            }

            # Update last_known_data with the latest prediction
            last_known_data = scaler_X.transform(prediction)

        # Convert to JSON
        predictions_json = json.dumps(predictions_dict, indent=4)

        print(f'predictions_json: {predictions_json}')

        return predictions_json

    def get_predictions(self, model_name):

        # Load the best model
        model1 = load_model(f'assets/models/{model_name}.keras')

        # Make predictions and plot results for training set
        train_predictions = model1.predict(self.X_train).flatten()
        train_result = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actual': self.y_train.flatten()})

        plt.plot(train_result['Train Predictions'], label='Train Predictions')
        plt.plot(train_result['Actual'], label='Actual')
        plt.legend()
        plt.title('Training Set Predictions vs Actual')
        # plt.show()

        # Make predictions and plot results for validation set
        val_predictions = model1.predict(self.X_val).flatten()
        val_result = pd.DataFrame(data={'Val Predictions': val_predictions, 'Actual': self.y_val.flatten()})

        plt.plot(val_result['Val Predictions'], label='Val Predictions')
        plt.plot(val_result['Actual'], label='Actual')
        plt.legend()
        plt.title('Validation Set Predictions vs Actual')
        # plt.show()

        # Make predictions and plot results for test set
        test_predictions = model1.predict(self.X_test).flatten()
        test_result = pd.DataFrame(data={'Test Predictions': test_predictions, 'Actual': self.y_test.flatten()})

        plt.plot(test_result['Test Predictions'], label='Test Predictions')
        plt.plot(test_result['Actual'], label='Actual')
        plt.legend()
        plt.title('Test Set Predictions vs Actual')
        # plt.show()

        # print(type(self.df_index))

        # Assume your test set's last date is the starting point for future predictions
        last_date = self.df_index[-1]  # Last date in the dataset

        # Number of future steps to predict
        future_steps = 12

        # Use the last window of data from your test set as the starting point
        last_window = self.X_test[-1]

        # Initialize a list to store future predictions
        future_predictions = []

        for _ in range(future_steps):
            # Predict the next value
            next_prediction = model1.predict(last_window[np.newaxis, :, :]).flatten()[0]
            future_predictions.append(next_prediction)

            # Update the window with the new prediction
            last_window = np.append(last_window[1:], [[next_prediction]], axis=0)

        # Inverse transform the predictions to original scale
        future_predictions_original_scale = self.scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)).flatten()

        # Generate corresponding future dates (monthly increments)
        future_dates = [last_date + relativedelta(months=i) for i in range(1, future_steps + 1)]

        # Convert predictions to float64 and create a dictionary of date-value pairs
        future_result = {str(date.date()): float(value) for date, value in
                         zip(future_dates, future_predictions_original_scale)}

        # Convert the dictionary to a JSON string
        json_future_results = json.dumps(future_result, indent=4)

        # Print the JSON formatted future predictions
        print(json_future_results)

    def get_dams_predictions(self, model_name, future_steps=30):
        X, y = self.df_to_X_y()

        # Split the data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Load the Keras model
        model = load_model(f'assets/models/{model_name}.keras')

        # Initialize last window
        last_window = X_test[-1]

        # Number of future steps
        future_predictions = []

        for _ in range(future_steps):
            next_prediction = model.predict(last_window[np.newaxis, :, :]).flatten()[0]
            future_predictions.append(next_prediction)
            last_window = np.append(last_window[1:], [[next_prediction]], axis=0)

        # Inverse transform
        future_predictions_original_scale = self.scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)).flatten()

        # Generate future dates
        last_date = self.df_index[-1]
        future_dates = [last_date + relativedelta(days=i) for i in range(1, future_steps + 1)]

        # Prepare results
        future_result = {str(date.date()): float(value) for date, value in
                         zip(future_dates, future_predictions_original_scale)}

        # Convert to JSON
        json_future_results = json.dumps(future_result, indent=4)
        print(json_future_results)

        # Make predictions and plot results for test set
        test_predictions = model.predict(X_test).flatten()
        test_result = pd.DataFrame(data={'Test Predictions': test_predictions, 'Actual': y_test.flatten()})

        plt.plot(test_result['Test Predictions'], label='Test Predictions')
        plt.plot(test_result['Actual'], label='Actual')
        plt.legend()
        plt.title('Test Set Predictions vs Actual')
        plt.show()

    def get_channel_predictions(self, model_name):
        X, y = self.df_to_X_y()

        # Split the data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Load the Keras model
        model = load_model(f'assets/models/{model_name}.keras')

        # Initialize last window
        last_window = X_test[-1]

        # Number of future steps
        future_steps = 12
        future_predictions = []

        for _ in range(future_steps):
            next_prediction = model.predict(last_window[np.newaxis, :, :]).flatten()[0]
            future_predictions.append(next_prediction)
            last_window = np.append(last_window[1:], [[next_prediction]], axis=0)

        # Inverse transform
        future_predictions_original_scale = self.scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)).flatten()

        # Generate future dates
        last_date = self.df_index[-1]
        future_dates = [last_date + relativedelta(months=i) for i in range(1, future_steps + 1)]

        # Prepare results
        future_result = {str(date.date()): float(value) for date, value in
                         zip(future_dates, future_predictions_original_scale)}

        # Convert to JSON
        json_future_results = json.dumps(future_result, indent=4)
        print(json_future_results)

        # Make predictions and plot results for test set
        test_predictions = model.predict(X_test).flatten()
        test_result = pd.DataFrame(data={'Test Predictions': test_predictions, 'Actual': y_test.flatten()})

        plt.plot(test_result['Test Predictions'], label='Test Predictions')
        plt.plot(test_result['Actual'], label='Actual')
        plt.legend()
        plt.title('Test Set Predictions vs Actual')
        plt.show()
