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
                   callbacks=[cp,es, lr], batch_size=32)


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
        future_dates = [last_date + relativedelta(days=i) for i in range(1, future_steps + 1)]

        # Convert predictions to float64 and create a dictionary of date-value pairs
        future_result = {str(date.date()): float(value) for date, value in
                         zip(future_dates, future_predictions_original_scale)}

        # Convert the dictionary to a JSON string
        json_future_results = json.dumps(future_result, indent=4)

        # Print the JSON formatted future predictions
        print(json_future_results)


