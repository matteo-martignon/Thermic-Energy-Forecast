def select_df_columns(df, col):
    '''
    Seleziona una lista di colonne.
    
    
    Params:
    -------
    df: pandas.dataframe
    col: list
        Lista di stringhe delle colonne da filtrare
        
    Return:
    -------
    pandas.dataframe
    
    '''
    return df[col]

def df_interpolate_and_dropna(df):
    '''
    Interpola i dati di un dataframe e cancella i rimanenti nulli
    
    '''
    return df.interpolate().dropna()
