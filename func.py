import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import pmdarima as pm
import time
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import warnings
import tensorflow as tf

warnings.filterwarnings("ignore")
import os

if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# plt.use('Agg')
def preprocess_data(file_path):
    data = read_csv(file_path, usecols=["DateTime", "Users"])
    data.dropna(how='any', axis=0, inplace=True)

    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data['DateTime'] = data['DateTime'].apply(lambda x: time.mktime(x.timetuple()))

    scaler = MinMaxScaler()
    data[['Users', 'DateTime']] = scaler.fit_transform(data[['Users', 'DateTime']])

    return data


def train_arima_model(train_data):
    model = pm.auto_arima(train_data['Users'], start_p=1, start_q=1,
                          information_criterion='aic',
                          test='adf',
                          max_p=8, max_q=8,
                          m=1,
                          d=0,
                          seasonal=False,
                          start_P=0,
                          D=0,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)

    return model


def forecast_arima(data_u, length):
    # model = ARIMA(data_u, order=model.order)
    model = ARIMA(data_u, order=(2, 0, 8))
    fit_model = model.fit()
    pred = fit_model.forecast(length)

    return pred


def evaluate_forecast(true_values, predicted_values):
    rmse = math.sqrt(mean_squared_error(true_values, predicted_values))
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)

    return rmse, mae, r2, mse


def plot_results(train_data, test_data, pred, plot_filename='arima.png'):
    plt.figure(figsize=(10, 6))

    plt.title('Users Over Time(Standardized)')
    plt.xlabel('DateTime')
    plt.ylabel('Users')
    plt.grid(True)
    plt.plot(train_data['DateTime'], train_data['Users'],
             label='training data', color='blue', linestyle='-')
    plt.plot(test_data['DateTime'], test_data['Users'],
             label='actual steam data', color='orange', linestyle='-')
    plt.plot(test_data['DateTime'], pred,
             label='prediction', color='red', linestyle='-')
    plt.legend()
    plt.savefig(plot_filename)


def init_data(path):
    df = pd.read_csv(path)
    df.index = df['DateTime'].astype('datetime64[ns]')
    df = df.drop(columns=['DateTime', 'Users Trend', 'In-Game'])
    df = df.dropna()

    return df


def LSTM_model_construct(input_size, output_size, df, scaler):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(100, input_shape=(input_size, 1), return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(80))
    model.add(tf.keras.layers.Dense(output_size, activation='tanh'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    df['Users'] = scaler.fit_transform(df['Users'].values.reshape(-1, 1))
    X_train, Y_train = [], []
    for i in range(0, int(df.shape[0] * 0.8) - input_size - output_size + 1):
        X_train.append(df['Users'][i: i + input_size].values.reshape(-1, 1))
        Y_train.append(df['Users'][i + input_size: i + input_size + output_size].values.reshape(-1, 1))
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    model.fit(X_train, Y_train, epochs=5, batch_size=16)

    return model


def forecast_LSTM(input_size, df, model):
    t = int(input_size * 12)
    Y_test = []
    for i in range(t, df.shape[0], 144):
        tmp = i
        for j in range(24):
            X_test = np.array([df['Users'][tmp - input_size: tmp].values.reshape(-1, 1)])
            Y_pred = model.predict(X_test).reshape(-1, 6)
            print(Y_pred)
            Y_test.append(Y_pred)
            tmp += 6
    Y_test = np.array(Y_test)
    return Y_test


def plot_LSTM(df, Y_test, scaler, plot_filename):
    X = range(df.shape[0])
    X_pred = []
    for i in range(1728, 2016):
        X_pred.append(i)
    plt.figure(figsize=(10, 8), dpi=150)
    plt.title('Users Over Time')
    plt.plot(X_pred, scaler.inverse_transform(Y_test.reshape(-1, 1)),
             label='prediction', color='red', linestyle='-')
    plt.plot(X, scaler.inverse_transform(df['Users'][:].values.reshape(-1, 1)),
             label='actual data', color='blue', linestyle='-')
    plt.legend()
    plt.savefig(plot_filename)
