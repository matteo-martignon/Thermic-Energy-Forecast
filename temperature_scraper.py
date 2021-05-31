import pandas as pd
from utils import get_soup, set_datetime_index

url_base = "https://www.3bmeteo.com/meteo/brescia/storico/"
years = list(range(2014, 2022))
months = ["{:0>2d}".format(x) for x in list(range(1, 13))]

l = []
for y in years:
    for m in months:
        year_month = f"{y}{m}"
        if year_month > '202105':
            break
        print(year_month)
        url = url_base + year_month
        html = get_soup(url)
        days = html.find_all("div", class_="col-xs-1-5 col-sm-1-7")

        for d in days:
            if d.find("span") == None:
                continue
            gg = int(d.find("strong").text)
            l.append({"day": f"{y}-{m}-{gg}",
                      "min": int(d.find("span").text.split()[0].replace('°C', '')),
                      "max": int(d.find("span", class_="arancio").text.replace('°C', ''))})

df = pd.DataFrame(l)
set_datetime_index(df, 'day')
df.to_csv('data/temperature_brescia.csv')
print('END')