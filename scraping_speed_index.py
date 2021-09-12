import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import copy
import os
from time import sleep
import re
import dask.dataframe as dd

# レースデータと対応をとるためのキー
common_head = ['year', 'date', 'month', 'kai', 'day',
               'place', 'sum_num', 'horse_num', 'horse_name']
# スピード指数
index_head = [#'horse_name',
              'lead_idx', '1st_lead_idx', '2nd_lead_idx', '3rd_lead_idx', '4th_lead_idx', '5th_lead_idx',
              'pace_idx', '1st_pace_idx', '2nd_pace_idx', '3rd_pace_idx', '4th_pace_idx', '5th_pace_idx',
              'rising_idx', '1st_rising_idx', '2nd_rising_idx', '3rd_rising_idx', '4th_rising_idx', '5th_rising_idx',
              'speed_idx', '1st_speed_idx', '2nd_speed_idx', '3rd_speed_idx', '4th_speed_idx', '5th_speed_idx', ]
# p_race_head = ['1st_place', '2nd_place', '3rd_place', '4th_place', '5th_place',
#                '1st_weather', '2nd_weather', '3rd_weather', '4th_weather', '5th_weather',
#                '1st_race_num', '2nd_race_num', '3rd_race_num', '4th_race_num', '5th_race_num',
#                '1st_sum_num', '2nd_sum_num', '3rd_sum_num', '4th_sum_num', '5th_sum_num',
#                '1st_horse_num', '2nd_horse_num', '3rd_horse_num', '4th_horse_num', '5th_horse_num',
#                '1st_rank', '2nd_rank', '3rd_rank', '4th_rank', '5th_rank',
#                '1st_field', '2nd_field', '3rd_field', '4th_field', '5th_field',
#                '1st_dist', '2nd_dist', '3rd_dist', '4th_dist', '5th_dist',
#                '1st_condi', '2nd_condi', '3rd_condi', '4th_condi', '5th_condi',
#                ]
# p_race_head = ['1st_place','1st_weather','1st_race_num','1st_sum_num','1st_horse_num','1st_rank','1st_field','1st_dist','1st_condi',
# '2nd_place','2nd_weather','2nd_race_num','2nd_sum_num','2nd_horse_num','2nd_rank','2nd_field','2nd_dist','2nd_condi',
# '3rd_place','3rd_weather','3rd_race_num','3rd_sum_num','3rd_horse_num','3rd_rank','3rd_field','3rd_dist','3rd_condi',
# '4th_place','4th_weather','4th_race_num','4th_sum_num','4th_horse_num','4th_rank','4th_field','4th_dist','4th_condi',
# '5th_place','5th_weather','5th_race_num','5th_sum_num','5th_horse_num','5th_rank','5th_field','5th_dist','5th_condi',
#                ]
p_race_head = ['1st_place', '1st_weather', '2nd_place', '2nd_weather', '3rd_place', '3rd_weather', '4th_place', '4th_weather', '5th_place', '5th_weather',
               '1st_field', '1st_dist', '1st_condi', '2nd_field', '2nd_dist', '2nd_condi', '3rd_field', '3rd_dist', '3rd_condi', '4th_field', '4th_dist', '4th_condi', '5th_field', '5th_dist', '5th_condi',
               '1st_sum_num', '1st_horse_num', '2nd_sum_num', '2nd_horse_num', '3rd_sum_num', '3rd_horse_num', '4th_sum_num', '4th_horse_num', '5th_sum_num', '5th_horse_num',
               '1st_rank', '2nd_rank', '3rd_rank', '4th_rank', '5th_rank',
               ]
# 1st_field, 1st_dist, 1st_condi, 2nd ..., 5th..., 1st_sumnum, 1st_horsenum,2nd...,5th...,1st_rank,2nd_rank

cols = ['year', 'date', 'month', 'race_num', 'field', 'dist', 'turn', 'weather',
        'field_cond', 'kai', 'day', 'place', 'sum_num', 'prize', 'rank',
        'horse_num', 'horse_name', 'sex', 'age', 'weight_carry', 'horse_weight',
        'weight_change', 'jockey', 'time', 'l_days', 'lead_idx', '1st_lead_idx',
        '2nd_lead_idx', '3rd_lead_idx', '4th_lead_idx', '5th_lead_idx',
        'pace_idx', '1st_pace_idx', '2nd_pace_idx', '3rd_pace_idx',
        '4th_pace_idx', '5th_pace_idx', 'rising_idx', '1st_rising_idx',
        '2nd_rising_idx', '3rd_rising_idx', '4th_rising_idx', '5th_rising_idx',
        'speed_idx', '1st_speed_idx', '2nd_speed_idx', '3rd_speed_idx',
        '4th_speed_idx', '5th_speed_idx', '1st_place', '1st_weather',
        '2nd_place', '2nd_weather', '3rd_place', '3rd_weather', '4th_place',
        '4th_weather', '5th_place', '5th_weather', '1st_field', '1st_dist',
        '1st_condi', '2nd_field', '2nd_dist', '2nd_condi', '3rd_field',
        '3rd_dist', '3rd_condi', '4th_field', '4th_dist', '4th_condi',
        '5th_field', '5th_dist', '5th_condi', '1st_sum_num', '1st_horse_num',
        '2nd_sum_num', '2nd_horse_num', '3rd_sum_num', '3rd_horse_num',
        '4th_sum_num', '4th_horse_num', '5th_sum_num', '5th_horse_num',
        '1st_rank', '2nd_rank', '3rd_rank', '4th_rank', '5th_rank']

last_race_column = ['レース名', 'コース', '騎手,斤量', '頭数,馬番,人気',
                    'タイム,(着順)', 'ﾍﾟｰｽ,脚質,上3F', '通過順位', 'ﾄｯﾌﾟ(ﾀｲﾑ差)', '馬体重()3F順', '先行指数', 'ペース指数', '上がり指数', 'スピード指数']
this_race_column = ['ﾍﾟｰｽ,脚質,上3F', '通過順位',
                    '馬体重()3F順', '先行指数', 'ペース指数', '上がり指数', 'スピード指数']
odds_column = ['着順']
fill_name = ['レース結果', '２走前の成績', '３走前の成績', '４走前の成績', '５走前の成績']


place_num = {'札幌': '01', '函館': '02', '福島': '03', '新潟': '04', '東京': '05',
             '中山': '06', '中京': '07', '京都': '08', '阪神': '09', '小倉': '10'}


def getSpeedIndex(url):
    res = requests.request("GET", url)

    # create BeatifulSoup object for scraping
    res.encoding = res.apparent_encoding
    # soup = BeautifulSoup(res.text, features='html.parser')
    soup = BeautifulSoup(res.content, features='html.parser')

    race_result_table_base = [tbody for tbody in soup.find_all('table', attrs={
                                                               'class': 'c1'})]
    race_result_table = race_result_table_base[0].find_all(
        'tr', recursive=False)

    merged = []
    fill_flg = False
    tmp_list = []
    for j, ee in enumerate(race_result_table):
        # 列数が出馬数+1なら配列を作成
        if len(ee.find_all('td', recursive=False)) == int(e['sum_num'])+1:
            row = []
            first = True
            # 馬名はテーブルが変なので別処理
            if ee.find_all('td', recursive=False)[int(e['sum_num'])].text == '馬名':
                for k in range(int(e['sum_num']), -1, -1):
                    if first:
                        row.append(ee.find_all('td', recursive=False)[k].text)
                        first = False
                    else:
                        row.append(ee.find_all('td', recursive=False)[k].find_all(
                            'td', {'class': 'c231'})[0].text.replace('ｌ', 'ー').replace('(外)', ''))
            else:  # 馬名以外の処理はまとめてやる
                for k in range(int(e['sum_num']), -1, -1):
                    if first and (ee.find_all('td', recursive=False)[k].text in fill_name):
                        fill_flg = True
                        if ee.find_all('td', recursive=False)[k].text == 'レース結果':
                            tmp_list = copy.copy(this_race_column)
                        elif ee.find_all('td', recursive=False)[k].text == '調教師':
                            tmp_list = copy.copy(odds_column)
                        else:
                            tmp_list = copy.copy(last_race_column)
                        first = False
                        row.append(ee.find_all('td', recursive=False)[k].text)
                    elif first and fill_flg:
                        try:
                            row.append(tmp_list.pop(0))
                        except IndexError:
                            fill_flg = False
                            row.append(ee.find_all(
                                'td', recursive=False)[k].text)
                        finally:
                            first = False
                    else:
                        row.append(ee.find_all('td', recursive=False)[k].text)
                        first = False
                for item in row:
                    if item:
                        break
                else:
                    continue

            merged.append(row)
        else:
            continue
    merged2 = np.array(merged).T.tolist()
    df = pd.DataFrame(data=np.array(
        merged2[1:]), columns=np.array(merged2[0]).T)
    return df


def getValue(a):
    if a.size > 0:
        return a[0]
    else:
        return ''


if __name__ == '__main__':
    df_allrace = pd.read_pickle('uma_prediction/dataset.pkl')
    url_prefix = 'http://jiro8.sakura.ne.jp/index.php?code='
    # df = pd.DataFrame(index=[], columns=cols)
    df = []
    for i, e in df_allrace.iterrows():
        param = e['year'][-2:]+place_num[e['place']] + \
            e['kai'].zfill(2)+e['day'].zfill(2)+str(e['race_num']).zfill(2)

        os.makedirs('uma_prediction/spdidx/'+e['year'], exist_ok=True)
        detail_path = 'uma_prediction/spdidx/'+e['year']+'/'+param+'.pkl'
        read_flg = False
        print('running... '+detail_path, end=':')
        if os.path.exists(detail_path):
            print(' loading')
            if not read_flg:
                df_detail = pd.read_pickle(detail_path)
                read_flg = True
        else:
            print(' scraping')
            sleep(1)
            df_detail = getSpeedIndex(url_prefix+param)
            df_detail.to_pickle(detail_path)
            read_flg = True
        # df_detail = df_detail[df_detail['馬名'].isin([e['horse_name']])]
        df_detail[df_detail['馬名'].str.contains(e['horse_name']+'$')]
        # df_index = df_detail[['馬名', '先行指数', 'ペース指数', '上がり指数', 'スピード指数']].set_axis(
        #     index_head, axis='columns')
        df_index = df_detail[['先行指数', 'ペース指数', '上がり指数', 'スピード指数']].set_axis(
             index_head, axis='columns')
        p_race_list = df_detail[['前走の成績', '２走前の成績', '３走前の成績', '４走前の成績',
                                 '５走前の成績', 'コース', '頭数,馬番,人気', 'タイム,(着順)']]
        p_race_list_ed = []

        for col, ee in p_race_list.iteritems():
            val = getValue(ee.values)
            if re.match(r'.+の成績$', col):
                # print('成績:'+val)
                p_race_list_ed.append(re.sub(r'\d+/\d+|\D$', '', val))  # 場所
                p_race_list_ed.append(re.sub(r'\d+/\d+\D', '', val))  # 天気
            elif re.match(r'コース', col):
                # print('コース:'+val)
                p_race_list_ed.append(re.sub(r'\d+\D', '', val))  # 芝 or ダート
                try:
                    p_race_list_ed.append(
                        int(re.sub(r'^\D|\D$', '', val)))  # 距離
                except:
                    p_race_list_ed.append(-1)
                p_race_list_ed.append(re.sub(r'\D\d+', '', val))  # 馬場状態
            elif re.match(r'頭数,馬番,人気', col):
                # print('出走馬:'+val)
                try:
                    p_race_list_ed.append(int(re.sub(r'ﾄ.+', '', val)))  # 馬数
                    p_race_list_ed.append(
                        int(re.sub(r'\d+ﾄ|番.*', '', val)))  # 馬番
                except:
                    p_race_list_ed.append(-1)
                    p_race_list_ed.append(-1)
            elif re.match(r'タイム,\(着順\)', col):
                # print('タイム:'+val)
                # p_race_list_ed.append(re.sub(r'\D$','',e)) # タイム
                try:
                    rank_char = re.sub(r'\d\.\d\d\.\d', '', val)
                    rank = ord(rank_char)-9311
                    p_race_list_ed.append(rank)  # 順位
                except:
                    p_race_list_ed.append(-1)  # 順位
        df.extend([e.T.tolist() + df_index.to_numpy().tolist()[0] + p_race_list_ed])

        # df_p_race = pd.DataFrame(data=np.array(
        #     [[e['horse_name']]+p_race_list_ed]), columns=['horse_name']+p_race_head)
        # df_index_all = pd.merge(df_index, df_p_race,
        #                         on='horse_name', how='right')
        # df_merged = pd.merge(e.to_frame().T, df_index_all,
        #                      on='horse_name', how='right')
        # df = df.append(df_merged, ignore_index=True)
    # print(df)
    df_last = pd.DataFrame(df, columns=cols)
    df_last.to_pickle('uma_prediction/index_dataset.pkl')
