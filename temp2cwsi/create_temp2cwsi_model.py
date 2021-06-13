import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
import lightgbm as lgbm
#LGBM 파라미터 설정
params={}
params['learning_rate']=0.02 #학습률 높을수록 학습 속도가 빠르나 정확도가 떨어질 수 있음
params['boosting_type']='gbdt' #GradientBoostingDecisionTree
params['objective']='regression' #Binary target feature
params['metric']='mse' #metric for binary classification
params['max_depth']=-1 # 무제한 깊이

if __name__=='__main__':
    try:
        data=pd.read_pickle('./data/not_cleansing_cwsi.pkl')
        features=['temp', 'ta_temp', 'ir_min', 'ir_max']
        X=data[features]
        y=data['target']
        print('data loading')
    except:
        data=get_train_data()
        data.to_pickle('./data/cleansing_data/not_cleansing_cwsi.pkl')
        print('data creating')
    print('split train test')
    print('train data shape : ',train_x.shape)
    print('test data shape : ', test_x.shape)
    train_x,test_x,train_y,test_y=train_test_split(X,y ,test_size=0.2,random_state=42)

    # 학습 데이터 분리 및 모델 학습
    train_ds = lgbm.Dataset(train_x, label = train_y) 
    test_ds = lgbm.Dataset(test_x, label = test_y)
    
    lgb = lgbm.train(params, train_ds, 5000, test_ds, verbose_eval=100, early_stopping_rounds=100)
    
    #joblib.dump(lgb,'temp2cwsi_15000ckp.pkl')