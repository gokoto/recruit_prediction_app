import numpy as np
import pandas as pd
import sklearn

import csv
train_x = pd.read_csv("./read_csv/train_x_3.csv", encoding='utf-8')
train_y = pd.read_csv("./read_csv/train_y.csv", encoding='utf-8')

y = train_y["応募数 合計"]
x = train_x

x = x.drop('勤務地　備考', axis=1)

x = x.fillna(0)

x = x.replace({"（紹介予定）雇用形態備考": {"正社員":2, "契約社員":1,"※ご紹介先により異なります。詳細はお問い合わせ下さい。":0,"パート社員":0,"アルバイト社員":0,"契約員":1,"契約契約社員":1}})

x_split_timecolumn = pd.concat([x, x['期間・時間\u3000勤務時間'].str.split('<BR>', expand=True)], axis=1).drop('期間・時間\u3000勤務時間', axis=1).drop(1, axis=1).drop(2, axis=1).drop(3, axis=1)

x = x_split_timecolumn.rename(columns={0: "期間・時間　勤務時間"})

x_split_timecolumn_2 = pd.concat([x, x['期間・時間\u3000勤務時間'].str.split('〜', expand=True)], axis=1).drop('期間・時間\u3000勤務時間', axis=1)

x = x_split_timecolumn_2.rename(columns={0: "勤務開始時間",1:"勤務終了時間"})

x['勤務開始時間'] = pd.to_datetime(x['勤務開始時間'])
x['勤務開始時間'] = x['勤務開始時間'].map(pd.Timestamp.timestamp)

x['勤務終了時間'] = pd.to_datetime(x['勤務終了時間'])
x['勤務終了時間'] = x['勤務終了時間'].map(pd.Timestamp.timestamp)

y_array = np.array(y)
x_array = np.array(x)

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(random_state=0)

rfr.fit(x_array, y_array)

