import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
# aqcuire netkeiba db data
# url = "https://db.netkeiba.com/race/202154071001/"

def getRaceDetail(url):
    res = requests.request("GET", url)

    # create BeatifulSoup object for scraping
    res.encoding = res.apparent_encoding
    soup = BeautifulSoup(res.text, features='html.parser')


    race_result_table_base = [tbody for tbody in soup.find_all('table',attrs={'class':'race_table_01'})]
    # pay_result_table_base = [tbody for tbody in soup.find_all('table',attrs={'class':'pay_table_01'})]
    race_result_table = race_result_table_base[0].find_all('tr')

    merged = []
    for i,e in enumerate(race_result_table):
        row = []
        if i == 0:
            for ee in e.find_all('th'):
                row.append(ee.text)
        else:
            for ee in e.find_all('td'):
                row.append(ee.text.replace('\n',''))
        merged.append(row)

    df = pd.DataFrame(data=np.array(merged[1:]),columns=merged[0])

    print(df)
    return df