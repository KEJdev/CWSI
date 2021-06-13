import os
import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#학습에 사용할 cwsi 데이터 경로
cwsi_path='../data/csv_cwsi/'
#학습에 사용할 metainfo 데이터 경로
ta_path='../data/csv_ta/metainfo_train_rev2.csv'
#학습에 사용할 resized 된 temp 데이터 경로
temp_path='../data/img_temp_resize/'

temp_list=os.listdir(temp_path)
cwsi_list=os.listdir(cwsi_path)
ta=pd.read_csv(ta_path)
features=['temp','ir_max','ir_min']


def get_clean_cwsi(cwsi):
    cwsi=np.array(cwsi,dtype=np.float64)
    cwsi[cwsi>1]=1
    cwsi[cwsi<0]=0
    return cwsi

#학습하기 위한 데이터 생성
def get_train_data():
    first=True
    for i,r in ta.iterrows():
        
        fn=str(int(r['name']))
        print('collecting data %s...'%(fn))
        
        temp_file_name=[filename for filename in temp_list if re.search(fn,filename)][0]
        temp=np.load(temp_path+temp_file_name)

        cwsi_file_name=[filename for filename in cwsi_list if  re.search(fn,filename)][0]
        cwsi=pd.read_csv(cwsi_path+cwsi_file_name,header=None)
        cwsi=cwsi.to_numpy()
        
        try:
            clean_cwsi=get_clean_cwsi(cwsi)
        except Exception as e:
            print(e)
            continue
        
        df=pd.DataFrame(temp.reshape(-1),columns=['temp'])
        df['target']=cwsi.reshape(-1)
        df.drop_duplicates(inplace=True)
        
        df['ta_temp']=r['temp']
        df['ir_min']=r['ir_min']
        df['ir_max']=r['ir_max']
        if first:
            data=df
            first=False
        else:
            data=pd.concat([data,df])
    return data
if __name__=='__main__':
    data=get_train_data()
    #데이터 저장
    #data.to_pickle('./data/not_cleansing_cwsi.pkl')