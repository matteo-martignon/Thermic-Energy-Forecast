import numpy as np
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def train_test_split(df, anno):
    train = df[:str(anno)].values
    test = df[str(anno+1):].values
    return train, test

def get_Xy_generator(x, lenght=24, batch_size=1):
    generator = TimeseriesGenerator(data=x, targets=x, length=lenght, batch_size=batch_size)
    X = []
    y = []
    for i, j in generator:
        X.append(i.flatten())
        y.append(j[0][0])
    X = np.array(X)
    y = np.array(y)
    
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    assert y.shape[0]==X.shape[0], "data leakage"
    
    return X, y

def build_LSTM_model(input_shape = 24):
    #Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(input_shape, 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    score = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f'mse = {mse}')
    print(f'mape = {mape}')
    print(f'r2_score = {score}')
