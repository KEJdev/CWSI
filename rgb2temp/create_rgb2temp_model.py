import numpy as np
import pandas as pd
from PIL import Image 
from sklearn.model_selection import train_test_split

if __name__=='__main__':
    #추출된 스케일 색상 데이터
    try:
        scalebar=pd.read_pickle('./data/scale_mean_rgb.pkl')
    except :
        scale_path='../data/img_scalebar/'
        from scale_bar_extractor import extract_scale_bar_numpy
        print('extract scale rgb data...')
        scalebar=extract_scale_bar_numpy(scale_path)

    #보라색 반점 부분 보완을 위한 추가 학습 데이터
    try:
        add_train_data=pd.read_csv('./data/add_train_data.csv')
        #컬럼 변경
        add_train_data.columns=[4,0,1,2,3]

        #기존 타깃 & 추가 타깃 데이터
        target=np.array(list(range(989,0,-1)))
        add_target=add_train_data[3]
        target=np.array(list(target) + add_target.tolist())
        #학습 데이터 통합
        scalebar=pd.concat([scalebar,add_train_data[[0,1,2]]],axis=0)
    except:
        scalebar=scalebar
        target=np.array(list(range(989,0,-1)))
    

    #학습, 검증 데이터 나누기
    train_features, test_features, train_labels, test_labels=scalebar,scalebar,target,target
    #train_features, test_features, train_labels, test_labels = train_test_split(scalebar, target, test_size = 0, random_state = 42)

    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    #모델 생성
    
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 47)
    #모델 학습
    rf.fit(train_features, train_labels)

    #모델 오차 확인
    
    predictions = rf.predict(test_features)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    #모델 저장
    # import pickle
    # filename = 'rgb2temp_natural2_addtrain.sav'
    # pickle.dump(rf, open(filename, 'wb'))