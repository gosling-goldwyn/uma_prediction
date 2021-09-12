import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
from time import sleep
from scraping_detail import getRaceDetail
import os

df_overview = pd.read_pickle(r"uma_prediction\pandas_obj.pkl")
print(df_overview)

head = ['year', 'date', 'month', 'race_num', 'field', 'dist', 'turn', 'weather', 'field_cond', 'kai', 'day', 'place',
        'sum_num', 'prize', 'rank', 'horse_num', 'horse_name', 'sex', 'age', 'weight_carry', 'horse_weight', 'weight_change', 'jockey', 'time', 'l_days']

# レースの回り順
counterclockwise = ['東京', '中京', '新潟']
clockwise = ['中山', '阪神', '京都', '札幌', '函館', '福島', '小倉']

url_prefix = "https://db.netkeiba.com"
all_race_data = []
for i, e in df_overview.iterrows():
    race_info = []
    if e['開催日'] == '開催日':
        continue
    # year
    race_info.append(int(e['開催日'].split('/')[0]))
    # date
    race_info.append(e['開催日'].split('/')[1]+'/'+e['開催日'].split('/')[2])
    # month
    race_info.append(int(e['開催日'].split('/')[1]))
    # race_num
    race_info.append(int(e['R']))
    # field
    race_info.append(re.sub(r'[0-9]+', '', e['距離']))
    # dist
    race_info.append(int(re.sub(r'\D', '', e['距離'])))
    # turn
    if e['開催'][1:-1] in clockwise:
        race_info.append('右')
    elif ['開催'][1:-1] in counterclockwise:
        race_info.append('左')
    else:
        race_info.append('不明')
    # weather
    race_info.append(e['天気'])
    # field_cond
    race_info.append(e['馬場'])
    # kai
    race_info.append(int(re.sub(r'\D+\d+','',e['開催'])))
    # day
    race_info.append(int(re.sub(r'\d+\D+','',e['開催'])))
    # place
    race_info.append(re.sub(r'^\d+','',re.sub(r'\d+$','',e['開催'])))
    

    # scraping race details
    os.makedirs('uma_prediction/races/', exist_ok=True)
    detail_path = 'uma_prediction/races/'+e['レース名'].replace('/','_')+'.pkl'
    if os.path.exists(detail_path):
        df_detail = pd.read_pickle(detail_path)
    else:
        df_detail = getRaceDetail(url_prefix+e['レース名'])
        df_detail.to_pickle(detail_path)
        sleep(1)

    # sum_num
    race_info.append(len(df_detail))
    
    for j, ee in df_detail.iterrows():
        race_info_detail = []
        race_info_detail += race_info
        # prize
        if ee['賞金(万円)']:
            race_info_detail.append(float(ee['賞金(万円)'].replace(',','')))
        else:
            race_info_detail.append(0)
        # rank
        if re.fullmatch(r'[0-9]+',ee['着順']):
            race_info_detail.append(int(ee['着順']))
        else:
            race_info_detail.append(-1)
        # horse_num
        race_info_detail.append(int(ee['馬番']))
        # horse_name
        race_info_detail.append(ee['馬名'])
        # sex
        race_info_detail.append(re.sub(r'[0-9]+', '', ee['性齢']))
        # age
        race_info_detail.append(int(re.sub(r'\D', '', ee['性齢'])))
        # weight_carry
        race_info_detail.append(float(ee['斤量']))
        # horse_weight
        try:
            race_info_detail.append(float(ee['馬体重'][:ee['馬体重'].find('(')-1]))
        except:
            race_info_detail.append(-1)
        # weight_change
        try:
            race_info_detail.append(float(ee['馬体重'][ee['馬体重'].find('('):].strip('()')))
        except:
            race_info_detail.append(0)
        # jockey
        race_info_detail.append(ee['騎手'])
        # time
        race_info_detail.append(ee['タイム'])
        # l_days
        race_info_detail.append('')
        all_race_data.append(race_info_detail)
df_all = pd.DataFrame(data=np.array(all_race_data).reshape(-1,len(head)),columns=head)
df_all.to_pickle('uma_prediction/dataset.pkl')
print(df_all)
