import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
from time import sleep

# aqcuire netkeiba db data

url = "https://db.netkeiba.com/"
serial = r"a%3A16%3A%7Bs%3A3%3A%22pid%22%3Bs%3A9%3A%22race_list%22%3Bs%3A4%3A%22word%22%3Bs%3A0%3A%22%22%3Bs%3A5%3A%22track%22%3Ba%3A2%3A%7Bi%3A0%3Bs%3A1%3A%221%22%3Bi%3A1%3Bs%3A1%3A%222%22%3B%7Ds%3A10%3A%22start_year%22%3Bs%3A4%3A%222010%22%3Bs%3A9%3A%22start_mon%22%3Bs%3A1%3A%221%22%3Bs%3A8%3A%22end_year%22%3Bs%3A4%3A%22none%22%3Bs%3A7%3A%22end_mon%22%3Bs%3A4%3A%22none%22%3Bs%3A3%3A%22jyo%22%3Ba%3A10%3A%7Bi%3A0%3Bs%3A2%3A%2201%22%3Bi%3A1%3Bs%3A2%3A%2202%22%3Bi%3A2%3Bs%3A2%3A%2203%22%3Bi%3A3%3Bs%3A2%3A%2204%22%3Bi%3A4%3Bs%3A2%3A%2205%22%3Bi%3A5%3Bs%3A2%3A%2206%22%3Bi%3A6%3Bs%3A2%3A%2207%22%3Bi%3A7%3Bs%3A2%3A%2208%22%3Bi%3A8%3Bs%3A2%3A%2209%22%3Bi%3A9%3Bs%3A2%3A%2210%22%3B%7Ds%3A9%3A%22kyori_min%22%3Bs%3A0%3A%22%22%3Bs%3A9%3A%22kyori_max%22%3Bs%3A0%3A%22%22%3Bs%3A4%3A%22sort%22%3Bs%3A4%3A%22date%22%3Bs%3A4%3A%22list%22%3Bs%3A3%3A%22100%22%3Bs%3A9%3A%22style_dir%22%3Bs%3A17%3A%22style%2Fnetkeiba.ja%22%3Bs%3A13%3A%22template_file%22%3Bs%3A14%3A%22race_list.html%22%3Bs%3A9%3A%22style_url%22%3Bs%3A18%3A%22%2Fstyle%2Fnetkeiba.ja%22%3Bs%3A6%3A%22search%22%3Bs%3A113%3A%22%B6%A5%C1%F6%BC%EF%CA%CC%5B%BC%C7%A1%A2%A5%C0%A1%BC%A5%C8%5D%A1%A2%B4%FC%B4%D6%5B2010%C7%AF1%B7%EE%A1%C1%CC%B5%BB%D8%C4%EA%5D%A1%A2%B6%A5%C7%CF%BE%EC%5B%BB%A5%CB%DA%A1%A2%C8%A1%B4%DB%A1%A2%CA%A1%C5%E7%A1%A2%BF%B7%B3%E3%A1%A2%C5%EC%B5%FE%A1%A2%C3%E6%BB%B3%A1%A2%C3%E6%B5%FE%A1%A2%B5%FE%C5%D4%A1%A2%BA%E5%BF%C0%A1%A2%BE%AE%C1%D2%5D%22%3B%7D"
payload=r'pid=race_list&word=&track%5B%5D=1&track%5B%5D=2&start_year=2010&start_mon=1&end_year=none&end_mon=none&jyo%5B%5D=01&jyo%5B%5D=02&jyo%5B%5D=03&jyo%5B%5D=04&jyo%5B%5D=05&jyo%5B%5D=06&jyo%5B%5D=07&jyo%5B%5D=08&jyo%5B%5D=09&jyo%5B%5D=10&kyori_min=&kyori_max=&sort=date&list=100'
headers = {
'Content-Type': 'application/x-www-form-urlencoded',
'Cookie': 'url=http%3A%2F%2Fdb.netkeiba.com%2F'
}


PAGENUM = 38480

merged = []
for i in range(1,PAGENUM//100):
    if i == 1:
        res = requests.request("POST", url, headers=headers, data=payload)
    else:
        payload = 'sort_key=date&sort_type=desc&page='+str(i)+'&serial='+serial+'&pid=race_list'
        res = requests.request("POST", url, headers=headers, data=payload)


    # create BeatifulSoup object for scraping
    soup = BeautifulSoup(res.content, features='html.parser',from_encoding='shift-jis')

    # get all race url
    l = [url.get('href') for url in soup.find_all('a') if re.match('\/race\/[0-9]+\/',url.get('href'))]
    print(l)

    race_result_table_base = [tbody for tbody in soup.find_all('table',attrs={'class':'race_table_01'})]
    race_result_table = race_result_table_base[0].find_all('tr')

    for i,e in enumerate(race_result_table):
        row = []
        if i == 0:
            for ee in e.find_all('th'):
                row.append(ee.text)
        else:
            for ee in e.find_all('td'):
                race_url = [url.get('href') for url in ee.find_all('a') if re.match('\/race\/[0-9]+\/',url.get('href'))]
                if race_url:
                    row.append(race_url[0])
                else:
                    row.append(ee.text.replace('\n',''))
        merged.append(row)
  
    sleep(1)
    # break

df = pd.DataFrame(data=np.array(merged[1:]),columns=merged[0])
df.to_pickle('uma_prediction/pandas_obj.pkl')
print(df)