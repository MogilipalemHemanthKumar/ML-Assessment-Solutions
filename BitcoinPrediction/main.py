from flask import Flask, render_template, request,send_file,url_for,redirect
import pandas as pd
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import io
import tensorflow as tf
from datetime import timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)




def load_dataset():
    maindf = pd.read_csv('./uploads/BTC-USD.csv')
    shape=maindf.shape
    closedf = maindf[['Date', 'Close']]
    closedf = closedf[closedf['Date'] > '2021-02-19']
    close_stock = closedf.copy()
    del closedf['Date']
    scaler = MinMaxScaler(feature_range=(0, 1))
    closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))
    training_size = int(len(closedf) * 0.60)
    test_size = len(closedf) - training_size
    train_data, test_data = closedf[0:training_size, :], closedf[training_size:len(closedf), :1]
    time_step = 15
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    return X_train,X_test,y_train,y_test,scaler

def model_build(X_train,X_test,y_train,y_test):
    model = Sequential()

    model.add(LSTM(10, input_shape=(None, 1), activation="relu"))

    model.add(Dense(1))

    model.compile(loss="mean_squared_error", optimizer="adam")
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=32, verbose=1)
    return model,history

def test(model,X_train,X_test,y_train,y_test,scaler):
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
    original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))
    train_rmse=math.sqrt(mean_squared_error(original_ytrain, train_predict))
    train_mse=mean_squared_error(original_ytrain, train_predict)
    train_mae=mean_absolute_error(original_ytrain, train_predict)
    test_rmse=math.sqrt(mean_squared_error(original_ytest, test_predict))
    test_mse=mean_squared_error(original_ytest, test_predict)
    test_mae=mean_absolute_error(original_ytest, test_predict)
    train_var=explained_variance_score(original_ytrain, train_predict)
    test_var=explained_variance_score(original_ytest, test_predict)
    train_r2=r2_score(original_ytrain, train_predict)
    test_r2=r2_score(original_ytest, test_predict)
    train_gamma=mean_gamma_deviance(original_ytrain, train_predict)
    test_gamma=mean_gamma_deviance(original_ytest, test_predict)
    train_poisson=mean_poisson_deviance(original_ytrain, train_predict)
    test_poisson=mean_poisson_deviance(original_ytest, test_predict)
    return train_rmse,train_mse,train_mae,test_rmse,test_mse,test_mae,train_var,test_var,train_r2,test_r2,train_gamma,test_gamma,train_poisson,test_poisson

def prediction(model,input_date,scaler):
    maindf = pd.read_csv('./uploads/BTC-USD.csv')
    maindf['Date'] = pd.to_datetime(maindf['Date'], format='%Y-%m-%d')
    date_15_days_before = input_date - timedelta(days=15)
    maindf = maindf.loc[(maindf['Date'] >= f'{date_15_days_before}')
                        & (maindf['Date'] < f'{input_date}')]
    closedf = maindf[['Date', 'Close']]
    del closedf['Date']
    closedf=scaler.transform(closedf)
    predicted_price = model.predict(closedf.reshape(1, -1, 1))
    return predicted_price

X_train, X_test, y_train, y_test, scaler = load_dataset()
model, history = model_build(X_train, X_test, y_train, y_test)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file and allowed_file(file.filename):
        file_contents = file.read()
        file.seek(0)  # Reset file pointer to the beginning
        df = pd.read_csv(StringIO(file_contents.decode('utf-8')))

        preview_html = df.head().to_html(classes='table table-striped', index=False)

        return render_template('preview.html', preview_table=preview_html)

    return 'File not allowed'


@app.route('/metrics_and_plot')
def show_metrics_and_plot():
    train_rmse, train_mse, train_mae, test_rmse, test_mse, test_mae, train_var, test_var, \
    train_r2, test_r2, train_gamma, test_gamma, train_poisson, test_poisson = test(model, X_train, X_test, y_train, y_test, scaler)

    metrics_data = {
        'Metric': ['RMSE', 'MSE', 'MAE', 'Variance', 'R2', 'Gamma', 'Poisson'],
        'Train': [train_rmse, train_mse, train_mae, train_var, train_r2, train_gamma, train_poisson],
        'Test': [test_rmse, test_mse, test_mae, test_var, test_r2, test_gamma, test_poisson]
    }

    metrics_df = pd.DataFrame(metrics_data)

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend(loc=0)

    # Save the plot to a file in the static folder
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f"static/plot_{timestamp}.png"
    plt.savefig(filename, format='png')
    plt.close()

    # Pass the filename to the template
    return render_template('metrics_and_plot.html',
                           metrics_table=metrics_df.to_html(classes='table table-striped', index=False),
                           plot_img=filename)


@app.route('/price',methods=['GET'])
def Home():
    return render_template('price.html')


@app.route('/predicted_price', methods=['POST'])
def predicted_price():
    if request.method == 'POST':
        input_date_str = request.form['date_input']
        input_date = datetime.strptime(input_date_str, '%Y-%m-%d')
        price=prediction(model,input_date,scaler)
        price=scaler.inverse_transform(price)
        return render_template('price.html', prediction_message=price)



if __name__ == '__main__':
    app.run()


