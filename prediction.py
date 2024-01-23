from flask import Flask, render_template
from functions.func import preprocess_data, train_arima_model, forecast_arima, \
    evaluate_forecast, plot_results, init_data, LSTM_model_construct, \
    forecast_LSTM, plot_LSTM
from sklearn.preprocessing import MinMaxScaler

prediction = Flask(__name__, static_url_path="/static")


@prediction.route('/')
def index():
    return render_template('index.html')


@prediction.route('/predict_arima')
def predict():
    # Replace 'your_data.csv' with the path to your CSV file
    data = preprocess_data('chart.csv')

    # Split data into training and testing sets
    train_data = data.iloc[:-144, :]
    test_data = data.iloc[-144:, :]

    # Train ARIMA model (uncomment if needed)
    # model = train_arima_model(train_data)

    # Make predictions
    # pred = forecast_arima(train_data['Users'], model, 144)
    pred = forecast_arima(train_data['Users'], 144)

    # Evaluate the forecast
    true_values = test_data['Users']
    rmse, mae, r2, mse = evaluate_forecast(true_values, pred)

    # Display results
    print(f'Test RMSE: {rmse:.2f}')
    print(f'Test MAE: {mae:.2f}')
    print(f'Test R2: {r2:.2f}')
    print(f'Test MSE: {mse:.3f}')

    plot_filename = './static/plot_arima.png'
    # Plot results
    plot_results(train_data, test_data, pred, plot_filename)

    return render_template('prediction_arima.html', rmse=rmse, mae=mae, r2=r2, mse=mse, plot_filename=plot_filename)


@prediction.route('/predict_lstm')
def index_LSTM():
    # Replace 'your_data.csv' with the path to your CSV file
    data = init_data('chart.csv')

    scaler = MinMaxScaler()

    # Train LSTM model
    model = LSTM_model_construct(144, 6, data, scaler)

    # Make predictions
    pred = forecast_LSTM(144, data, model)

    # Evaluate the forecast
    true_values = data['Users'][1727:data.shape[0]]
    rmse, mae, r2, mse = evaluate_forecast(true_values, pred.reshape(-1, 1))

    # Display results
    print(f'Test RMSE: {rmse:.2f}')
    print(f'Test MAE: {mae:.2f}')
    print(f'Test R2: {r2:.2f}')
    print(f'Test MSE: {mse:.3f}')

    plot_filename = './static/plot_lstm.png'
    # Plot results
    plot_LSTM(data, pred, scaler, plot_filename)

    return render_template('prediction_lstm.html', rmse=rmse, mae=mae, r2=r2, mse=mse, plot_filename=plot_filename)


if __name__ == '__main__':
    prediction.run(debug=False)
