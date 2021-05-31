import pandas as pd
import requests
from bs4 import BeautifulSoup

def clear_month(data):
    '''
    Sostituisce il mese in formato stringa con il mese in formato numerico all'interno di una stringa.
    
    Params:
    -------
    data: str
        Deve essere in formato gg-MMM-yy, es. 01-GEN-18
    
    Return:
    -------
    data: str
    
    '''
    d = {'GEN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAG': '05', 'GIU': '06', 'LUG': '07', 'AGO': '08', 'SET': '09',
       'OTT': '10', 'NOV': '11', 'DIC': '12'}
    for k, v in d.items():
        if data.split('-')[1] == k:
            data = data.replace(k,v)
    return data

def set_datetime_index(df, column):
    '''
    Imposta la colonna di un dataframe pandas come indice datetime.
    
    Params:
    -------
    df: pandas.dataframe
        Dataframe contenente column
    column: str, contenente una data
        Colonna di df da impostare come indice datetime
    
    Return:
    -------
    None
    
    '''
    df[column] = pd.to_datetime(df[column])
    df.set_index(column, inplace=True)
    return None

def df_datetimeindex_resampleHmean(df, column):
    '''
    Imposta la colonna di un dataframe pandas come indice datetime e fa il resample con frequenza oraria usando la media.
    
    Params:
    -------
    df: pandas.dataframe
        Dataframe contenente column
    column: str, contenente una data
        Colonna di df da impostare come indice datetime
    
    Return:
    -------
    pandas.dataframe
    
    '''
    set_datetime_index(df, column)
    return df.resample('H').mean()

def clear_columns(columns):
    '''
    Pulisce i nomi colonna secondo il dizionario definito nella funzione
    '''
    dict_col_clear = {"Radiazione (W/m2)": "radiazione(W/m2)",
    "Consuntivo_radiazione": "consuntivo_radiazione",
    "TOTALLOADVALUE": "total_load_value",
    "DomandaElettrica": "domanda_elettrica",
    "ET Rete": "ET_rete",
    "PotenzaTermicaOraria": "potenza_termica_oraria",
    "RADIAZIONE (W/m2)": "radiazione(W/m2)",
    "Previsioni_radiazione": "previsioni_radiazione",
    "Precipitazioni (mm)": "precipitazioni(mm)",
    "Pressione (bar)": "pressione(bar)",
    "Umidita (%)": "umidita(%)",
    "Temperatura": "temperatura", 
    "Previsioni_Temperatura": "previsioni_temperatura"}
    
    columns = list(columns)
    l = []
    for column in columns:
        for k, v in dict_col_clear.items():
            column = column.replace(k,v)
        l.append(column)
    return l

def process_PrevisioniTemperatura(df):
    df_temp = df.drop(columns=["IdPrevisione","FornitorePrevisioni","DataEmissione"]).melt(id_vars=["DataPrevisione"], 
        var_name="Orarii", 
        value_name="Temperatura")
    df_temp["Orarii"] = df_temp["Orarii"].apply(lambda x: x[4:6])
    df_temp["datetime"] = df_temp["DataPrevisione"]+ ' ' +df_temp["Orarii"]
    df_temp["datetime"] = pd.to_datetime(df_temp["datetime"], format="%d/%m/%Y %H")
    df_temp = df_temp.set_index("datetime")[["Temperatura"]].loc["2014":].resample('H').last()
    return df_temp

def get_data(path):
    '''
    Carica i dati nei file definiti nella lista datasets, li processa e li unisce in un unico pandas.dataframe.
    
    Params:
    -------
    path: str
        Path dove sono i file
    
    Return:
    -------
    df_merge: pandas.dataframe
        Dataframe con tutti i file uniti
    df: dict
        dizionario di tutti i dataframe contenenti i file
    
    '''
    df = {} 
    dt_columns = {"Consuntivo_radiazione": "Date-Time","DomandaElettrica": "DATETIME","PotenzaTermicaOraria": "Orario",
                  "Previsioni_radiazione": "ORARIO","consuntivi_meteo": "Date-Time"}
    datasets = ["Consuntivo_radiazione","DomandaElettrica","PotenzaTermicaOraria","PrevisioniTemperatura"
                ,"Previsioni_radiazione","consuntivi_meteo"] 

    for nome in datasets:
        df[nome] = pd.read_csv(f"{path}/{nome}.csv", sep =";", encoding='latin1', decimal = ",", parse_dates=True)

    for k, v in dt_columns.items():
        if k=="Previsioni_radiazione":
            df[k][v] = df[k][v].apply(clear_month)
        df[k] = df_datetimeindex_resampleHmean(df[k], v)
        if k=="DomandaElettrica":
            df[k] = df[k][["TOTALLOADVALUE"]]
        df[k].columns = df[k].columns + f" ({k})"
        df[k].columns = clear_columns(df[k].columns)
    
    df["PrevisioniTemperatura"] = process_PrevisioniTemperatura(df["PrevisioniTemperatura"])
    df["PrevisioniTemperatura"].columns = df["PrevisioniTemperatura"].columns + " (Previsioni_Temperatura)"
    df["PrevisioniTemperatura"].columns = clear_columns(df["PrevisioniTemperatura"].columns)
    
    for k in datasets:
        if 'df_merge' not in locals():
            df_merge = df[k]
        else:
            df_merge = df_merge.merge(df[k], how='outer', left_index=True, right_index=True)
    return df_merge, df

def get_soup(url):
    response = requests.get(url)
    return BeautifulSoup(response.content, features='html.parser')

def get_temperature_data(path):
    df_temp = pd.read_json(path, orient = 'index')
    df_temp.index = pd.to_datetime(df_temp.index)
    df_temp = df_temp.resample("H").last().fillna(method="ffill")
    
    df_temp.columns = ["temp_min", "temp_max"]
    df_temp["temp_media"] = df_temp[["temp_min", "temp_max"]].sum(axis=1)/2
    
    return df_temp
