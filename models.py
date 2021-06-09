import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from prophet.serialize import model_from_json
import json
from joblib import load
from fbprophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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
    model.add(LSTM(128, return_sequences=False, input_shape=(input_shape, 1)))
    #model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def evaluate_model(y_true, y_pred):
    '''
    Params:
    -------
    y_true: numpy.array
    y_pred: numpy.array
    
    Returns:
    --------
    None
    
    '''
    mse = mean_squared_error(y_true, y_pred)
    score = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f'mse = {mse}')
    print(f'r2_score = {score}')
    print(f'mape = {mape}')
    print(f'mae = {mae}')
    return None
    
def get_forecast_measures(y_true, y_pred):
    '''
    Params:
    -------
    y_true: numpy.array
    y_pred: numpy.array
    
    Returns:
    --------
    d: dict
    
    '''
    d = {}
    for i in range(1, len(y_true)+1):
        d[i] = {"mae" : mean_absolute_error(y_true[:i], y_pred[:i]),
                "mape" : mean_absolute_percentage_error(y_true[:i], y_pred[:i])}
    return d

def get_model_predictions(train, test, model, window = 24):
    """
    Create prediction for RNN model fitted with generator
    
    Parameters
    ----------
    train : np.array
        training scaled data
    test : np.array | pd.DataFrame
        test data
    model : keras.model
        keras RNN model
    
    Returns
    -------
    list
        model predictions
    """
    y_hat = []
    batch = train[-window:].reshape((1, window, 1))
    for i in range(len(test)):
        # si calcola la previsione 1 mese in avanti 
        # ([0] è per recuperare il valore numerico al posto dell'intero array)
        new_pred = model.predict(batch)[0][0]
        # salvataggio previsioni
        y_hat.append(new_pred)
        # aggiornamento batch per includere le previsioni ed eliminare il primo valore
        batch = np.append(arr=batch[:, 1:, :], values=[[[new_pred]]], axis=1)
    return y_hat

def stan_init(m):
    """Retrieve parameters from a trained model.
    
    Retrieve parameters from a trained model in the format
    used to initialize a new Stan model.
    
    Parameters
    ----------
    m: A trained model of the Prophet class.
    
    Returns
    -------
    A Dictionary containing retrieved parameters of m.
    
    """
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        res[pname] = m.params[pname][0][0]
    for pname in ['delta', 'beta']:
        res[pname] = m.params[pname][0]
    return res

def load_prophet_model(path):
    with open(path, 'r') as fin:
        model = model_from_json(json.load(fin))
    return model

def load_scaler(path):
    return load(path)

def get_model_metrics(y_true, y_pred):
    '''
    Params:
    -------
    y_true: numpy.array
    y_pred: numpy.array
    
    Returns:
    --------
    mae:
    mape:
    
    '''
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mae, mape

def model_add_observations(model, X_train):
    '''
    Params:
    -------
    model: Prophet model
        model to update
    X_train: pandas.dataframe or pandas.series
        X_train must have 1 column: univariate model
    
    Return:
    -------
    new_model: Prophet model
    '''
    df_train_prophet = X_train.reset_index()
    df_train_prophet.columns = ['ds', 'y']
    new_model = Prophet().fit(df_train_prophet, init=stan_init(model))
    return new_model

def get_prophet_predictions(model, scaler=None, periods=48, freq='H'):
    future = model.make_future_dataframe(periods=periods, freq=freq, include_history=False)
    test_predictions = model.predict(future)
    if scaler is None:
        return test_predictions['yhat'].values
    else:
        return scaler.inverse_transform(test_predictions['yhat'].values)
    
def X_train_add_observations(train, observations, scaler=None):
    X_train = train.copy()
    X_add = observations.copy()
    if scaler is None:
        return X_train.append(X_add)
    else:
        X_add.loc[:, :] = scaler.transform(X_add.values)
        X_train.loc[:, :] = scaler.transform(X_train.values)
        return X_train.append(X_add)
        
def get_y_test_new(X_train_new, df_test, periods=24):
    return df_test.loc[X_train_new.index[-1]:].iloc[1:periods+1].values


def get_prophet_predictions_dataframe(model_path, periods=48, scaler=None, results_to_file=False):
    '''
    Ritorna un dataframe di predizioni a partire da un file contenente un modello prophet
    '''
    model = load_prophet_model(model_path)
    y_pred = get_prophet_predictions(model=model, periods=periods, scaler=None)

    start = model.history_dates.iloc[-1] + pd.Timedelta(hours=1)

    data = pd.DataFrame(y_pred, index=pd.date_range(start=start, periods=len(y_pred), freq='H'), columns=["predictions"])
    
    if results_to_file==True:
        from datetime import datetime
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        data.to_csv(f"{now}_predictions.csv")
        return data
    else:
        return data

def get_EXPO_model(X_train):
    model = ExponentialSmoothing(X_train,
                            trend="add",
                            damped_trend=False,
                            seasonal="add",
                            seasonal_periods=24,
                            initialization_method=None,
                            initial_level=None,
                            initial_trend=None,
                            initial_seasonal=None,
                            use_boxcox=None,
                            bounds=None,
                            dates=None,
                            freq="H",
                            missing='none')
    model_fit = model.fit(
                    smoothing_level=0.5,
                    smoothing_trend=None,
                    smoothing_seasonal=0.5,
                    damping_trend=None,
                    optimized=True,
                    remove_bias=False,
                    start_params=None,
                    method=None,
                    minimize_kwargs=None,
                    use_brute=True,
                    use_boxcox=None,
                    use_basinhopping=None,
                    initial_level=None,
                    initial_trend=None)
    return model_fit


