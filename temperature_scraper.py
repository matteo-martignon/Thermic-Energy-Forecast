import json
from utils import get_soup

url_base = "https://www.3bmeteo.com/meteo/brescia/storico/"
years = list(range(2014, 2022))
months = ["{:0>2d}".format(x) for x in list(range(1, 13))]

data = {}
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
            if d.find("span") is None:
                continue
            gg = int(d.find("strong").text)
            data[f"{y}-{m}-{gg}"] = {"min": int(d.find("span").text.split()[0].replace('°C', '')),
                                     "max": int(d.find("span", class_="arancio").text.replace('°C', ''))}

with open('data/temperature_brescia.json', 'w') as f:
    json.dump(data, f)
