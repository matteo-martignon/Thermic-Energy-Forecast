
import pandas as pd

from models import evaluate_model, load_prophet_model,get_model_metrics, model_add_observations, get_prophet_predictions, X_train_add_observations, get_y_test_new

from utils import get_ETRete_clear


def get_prophet_evaluation(model_path, train_path, test_path, window=48, step_forward=24, verbose=True):
    model = load_prophet_model(model_path)
    df_train = get_ETRete_clear(train_path)
    df_test = get_ETRete_clear(test_path)
    col = df_train.columns[0]
    
    d = {}
    for i in range(1, int(df_test.shape[0]/window)):
        
        if i==1:
            end_window = i*window
            y_true = df_test[col].iloc[:end_window].values
            y_pred = get_prophet_predictions(model, periods=window)
            mae, mape = get_model_metrics(y_true, y_pred)
            d[i] = {"mae": mae, "mape": mape}
            if verbose==True:
                print("step: ", i)
                #print(y_true, y_pred)
                print("mae: ", mae,"\t mape: ", mape)
        elif i==2:
            end_window+=step_forward
            X_train_new = X_train_add_observations(df_train, df_test[[col]].iloc[:end_window])
            new_model = model_add_observations(model, X_train_new)
            y_pred_new = get_prophet_predictions(new_model, periods=window)
            y_true_new = get_y_test_new(X_train_new, df_test, periods=window)
            mae, mape = get_model_metrics(y_true_new, y_pred_new)
            d[i] = {"mae": mae, "mape": mape}
            if verbose==True:
                print("step: ", i)
                #print(X_train_new)
                #print(y_true, y_pred)
                print("mae: ", mae,"\t mape: ", mape)
        else:
            end_window+=step_forward
            X_train_new = X_train_add_observations(df_train, df_test[[col]].iloc[:end_window])
            new_model = model_add_observations(new_model, X_train_new)
            y_pred_new = get_prophet_predictions(new_model, periods=window)
            y_true_new = get_y_test_new(X_train_new, df_test, periods=window)
            mae, mape = get_model_metrics(y_true_new, y_pred_new)
            d[i] = {"mae": mae, "mape": mape}
            if verbose==True:
                print("step: ", i)
                #print(X_train_new)
                #print(y_true, y_pred)
                print("mae: ", mae,"\t mape: ", mape)

    return d


