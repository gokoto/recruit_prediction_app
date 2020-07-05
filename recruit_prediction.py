import os
from flask import Flask, flash, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import sklearn
import csv
import data_shaping

DOWNLOAD_FOLDER = './downloads'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prediction(test_x):
    test_x = pd.read_csv(test_x)
    test_x_jobNo = test_x[["大手企業", "1日7時間以下勤務OK", "派遣形態", "正社員登用あり", "残業月20時間未満", "仕事の仕方", "未経験OK", "（紹介予定）雇用形態備考", "期間・時間　勤務時間", 'お仕事No.']]
    test_x = test_x_jobNo.drop("お仕事No.", axis=1)
    test_x = test_x.fillna(0)
    test_x = test_x.replace({"（紹介予定）雇用形態備考": {"正社員":2, "契約社員":1,"※ご紹介先により異なります。詳細はお問い合わせ下さい。":0,"パート社員":0,"アルバイト社員":0,"契約員":1,"契約契約社員":1}})
    test_x_split_timecolumn = pd.concat([test_x, test_x['期間・時間\u3000勤務時間'].str.split('<BR>', expand=True)], axis=1).drop('期間・時間\u3000勤務時間', axis=1).drop(1, axis=1).drop(2, axis=1).drop(3, axis=1)
    test_x = test_x_split_timecolumn.rename(columns={0: "期間・時間　勤務時間"})
    test_x_split_timecolumn_2 = pd.concat([test_x, test_x['期間・時間\u3000勤務時間'].str.split('〜', expand=True)], axis=1).drop('期間・時間\u3000勤務時間', axis=1)
    test_x = test_x_split_timecolumn_2.rename(columns={0: "勤務開始時間",1:"勤務終了時間"})
    test_x['勤務開始時間'] = pd.to_datetime(test_x['勤務開始時間'])
    test_x['勤務開始時間'] = test_x['勤務開始時間'].map(pd.Timestamp.timestamp)
    test_x['勤務終了時間'] = pd.to_datetime(test_x['勤務終了時間'])
    test_x['勤務終了時間'] = test_x['勤務終了時間'].map(pd.Timestamp.timestamp)
    y_pred = data_shaping.rfr.predict(test_x)
    y_pred = pd.DataFrame(y_pred, columns=["応募数 合計"])
    y_pred['お仕事No.'] = test_x_jobNo['お仕事No.']
    y_pred = y_pred[['お仕事No.','応募数 合計']]
    y_pred_csv = y_pred.to_csv('./downloads/y_pred.csv', index=False)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if (os.path.exists('./downloads/y_pred.csv')):
       os.remove('./downloads/y_pred.csv')
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            prediction(file)

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    <a href="/data/download">download</a><br>
    '''

@app.route('/data/download')
def download():
    return send_file('./downloads/y_pred.csv',
                     mimetype='text/csv',
                     attachment_filename='y_pred.csv',
                     as_attachment=True)
