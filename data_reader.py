from utils import get_data, get_temperature_data

df, df_raw = get_data('./data')
df_temp = get_temperature_data("./data/temperature_brescia.json")
df = df_temp.merge(df, how='outer', left_index=True, right_index=True)

print(df.head())
