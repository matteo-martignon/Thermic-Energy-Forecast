from utils import get_data, get_temperature_data
from preprocessing import select_df_columns, df_interpolate_and_dropna
from models import train_test_split, build_LSTM_model, get_Xy_generator, evaluate_model

from sklearn.preprocessing import StandardScaler
ssc = StandardScaler()

from warnings import filterwarnings
filterwarnings("ignore")

from datetime import datetime

print("Importing data...")
df, df_raw = get_data('./data')
df_temp = get_temperature_data("./data/temperature_brescia.json")
df = df_temp.merge(df, how='outer', left_index=True, right_index=True)

print("Start preprocessing data...")
df_prep = df.pipe(select_df_columns, ["ET_rete (potenza_termica_oraria)"])\
    .pipe(df_interpolate_and_dropna)

train, test = train_test_split(df_prep, 2019)

train = ssc.fit_transform(train)
test = ssc.transform(test)

X_train, y_train = get_Xy_generator(train)
X_test, y_test = get_Xy_generator(test)

model = build_LSTM_model()

print("Start training model...")
model.fit(X_train, y_train, epochs=5, validation_split=0.2)

y_pred = model.predict(X_test)

y_test = ssc.inverse_transform(y_test.reshape(-1,1))
y_pred = ssc.inverse_transform(y_pred.reshape(-1,1))

evaluate_model(y_test, y_pred)

model.evaluate(X_test, ssc.transform(y_test.reshape(-1,1)))

now = datetime.utcnow().strftime("%Y%m%d%H%M")
model.save(f'models/{now}_LSTM1.h5')

print("END, model saved")
