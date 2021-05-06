import pickle
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import cv2
import os
import re
from sklearn.metrics import mean_squared_error

class Converter_rgb2temp():
    
    def __init__(self,model_path='./rgb2temp/rgb2temp_model.sav'):
        self.model = pickle.load(open(model_path, 'rb'))
        return 
    
    def convert_rgb2temp(self,rgb,ir_min,ir_max):
        img_array=np.array(rgb).reshape(-1,3)
        predict=self.model.predict(img_array)
        scaled_temp=(ir_max-ir_min)/990
        temp=predict*scaled_temp + ir_min
        temp=temp.round(2)
        return temp.reshape(rgb.shape[:2])

    
class Converter_temp2cwsi():
    
    def __init__(self,model_path='./temp2cwsi/temp2cwsi_15000ckp.pkl',output_size=(347,435)):
        self.model = joblib.load(model_path)
        self.output_size=output_size
        return
    
    def clean_data(self,data):
        data=np.array(data,dtype=np.float64)
        data[data>1]=1
        data[data<0]=0
        return data
    
    def convert_temp2cwsi(self,temp,ta_temp,ir_min,ir_max):
        df=pd.DataFrame(temp.reshape(-1),columns=['temp'])
        df['ta_temp']=ta_temp
        df['ir_min']=ir_min
        df['ir_max']=ir_max
        predict = self.model.predict(df)
        result=predict.reshape(self.output_size)
        clean_result=self.clean_data(result)
        return clean_result
    

    
class imgResizer():
    
    def __init__(self,output_size=(347,435)):
        self.output_size=output_size
        return
    
    def img_resize(self,img):
        #축소 기법중에선 area를 많이 씀
        resize = cv2.resize(img, dsize=(self.output_size[1],self.output_size[0]), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        if img.shape[-1]==3:
            resize=self.bgr2rgb(resize)
            
        return resize

    def bgr2rgb(self,img):
        b, g, r = cv2.split(img)   # img파일을 b,g,r로 분리
        img2 = cv2.merge([r,g,b]) # b, r을 바꿔서 Merge
        return img2
    

class dataLoader():
    
    def __init__(self,data_type):
        self.data_type=data_type
        return
    
    def load(self,path):
        if self.data_type=='csv':
            data=pd.read_csv(path,header=None)
        elif self.data_type=='npy':
            data=np.load(path)
        else :
            data=cv2.imread(path, cv2.IMREAD_COLOR)
        return data


    
class error_Calculator():

    def clean_data(self,data):
        data=np.array(data,dtype=np.float64)
        data[data>1]=1
        data[data<0]=0
        return data
    
    def mean_square_error(self,pred,target):
        clean_target=self.clean_data(target)
        error=mean_squared_error(pred.reshape(-1), clean_target.reshape(-1))
        return error

    
    
def predict_cwsi(config):
    #모델이 저장된 경로
    temp_model_path=config['path']['temp_model_path']
    rgb_model_path=config['path']['rgb_model_path']

    #인풋 데이터 저장된 디렉토리
    input_path=config['path']['input_path']
    #추가 데이터 ir_min, ir_max, ta_temp 저장 디렉토리
    ta_path=config['path']['ta_path']

    #만약 처음 데이터가 온도면 Architecture='temp2cwsi'
    #              rgb데이터면 Architecture='rgb2cwsi'
    Architecture=config['model_parameter']['Architecture']

    #결과 cwsi를 저장할 디렉토리
    result_path=config['path']['result_path']

    #결과 cwsi의 사이즈
    output_size=eval(config['model_parameter']['output_size'])

    #정답 데이터 저장 디렉토리
    target_path=config['path']['target_path']

    if Architecture=='rgb2cwsi':
        print('process Architecture1 - rgb2cwsi')
        print('rgb model loading...')
        rgb_converter=Converter_rgb2temp(rgb_model_path)
    else:
        print('process Architecture2 - temp2cwsi')


    print('temp model loading...')
    temp_converter=Converter_temp2cwsi(temp_model_path,output_size)

    img_Resizer=imgResizer(output_size)

    data_list=os.listdir(input_path)

    data_type=data_list[5].split('.')[-1]

    Loader=dataLoader(data_type)

    if target_path:
        target_list=os.listdir(target_path)
        calculator=error_Calculator()

    ta=pd.read_csv(ta_path)

    for i,r in ta.iterrows():
        fn=str(int(r['name']))
        print('=================start %s file predict================='%(fn))
        try:
            file_name=[filename for filename in data_list if re.search(fn,filename)][0]
        except exception as e:
            print(e)
            continue
        data_path=input_path+file_name
        data=Loader.load(data_path)

        ta_temp=r['temp']
        ir_min=r['ir_min']
        ir_max=r['ir_max']
        
        if data.shape[:2]==output_size:
            print('equal data size : ', data.shape[:2], output_size)
        else:
            print('data resize : %s -> %s'%(data.shape[:2], output_size))
            data=img_Resizer.img_resize(data)
        
        if Architecture=='rgb2cwsi':
            #data, ir_min, ir_max
            print('convert rgb 2 temp...')
            data=rgb_converter.convert_rgb2temp(data,ir_min,ir_max)
        else:
            data=data
            
        #temp, ta_temp,ir_min,ir_max
        print('convert temp 2 cwsi...')
        cwsi=temp_converter.convert_temp2cwsi(data,ta_temp,ir_min,ir_max)
        


        if target_path:
            print('calculate error...')
            try:
                file_name=[filename for filename in target_list if re.search(fn,filename)][0]
            except exception as e:
                print(e)
                continue
            target=pd.read_csv(target_path+file_name,header=None)
            error=calculator.mean_square_error(cwsi,target)
            print('%s mean square error :'%(fn), error)

        pd.DataFrame(cwsi).to_csv(result_path+fn+'.csv')
        
        if i > 1:
            break
            
def predict_cwsi2(config):
    #모델이 저장된 경로
    temp_model_path=config['path']['temp_model_path']
    rgb_model_path=config['path']['rgb_model_path']

    #인풋 데이터 저장된 디렉토리
    input_path=config['path']['input_path']
    #추가 데이터 ir_min, ir_max, ta_temp 저장 디렉토리
    ta_path=config['path']['ta_path']

    #만약 처음 데이터가 온도면 Architecture='temp2cwsi'
    #              rgb데이터면 Architecture='rgb2cwsi'
    Architecture=config['model_parameter']['Architecture']

    #결과 cwsi를 저장할 디렉토리
    result_path=config['path']['result_path']

    #결과 cwsi의 사이즈
    output_size=eval(config['model_parameter']['output_size'])

    #정답 데이터 저장 디렉토리
    target_path=config['path']['target_path']

    if Architecture=='rgb2cwsi':
        print('process Architecture1 - rgb2cwsi')
        print('rgb model loading...')
        rgb_converter=Converter_rgb2temp(rgb_model_path)
    else:
        print('process Architecture2 - temp2cwsi')


    print('temp model loading...')
    temp_converter=Converter_temp2cwsi(temp_model_path,output_size)

    img_Resizer=imgResizer(output_size)

    data_list=os.listdir(input_path)

    data_type=data_list[5].split('.')[-1]

    Loader=dataLoader(data_type)

    if target_path:
        target_list=os.listdir(target_path)
        calculator=error_Calculator()

    ta=pd.read_csv(ta_path)

    for i,r in ta.iterrows():
        fn=str(int(r['name']))
        print('=================start %s file predict================='%(fn))
        try:
            file_name=[filename for filename in data_list if re.search(fn,filename)][0]
        except exception as e:
            print(e)
            continue
        data_path=input_path+file_name
        data=Loader.load(data_path)

        ta_temp=r['temp']
        ir_min=r['ir_min']
        ir_max=r['ir_max']
        
        if Architecture=='rgb2cwsi':
            #data, ir_min, ir_max
            print('convert rgb 2 temp...')
            data=rgb_converter.convert_rgb2temp(data,ir_min,ir_max)
        else:
            data=data
            
        if data.shape[:2]==output_size:
            print('equal data size : ', data.shape[:2], output_size)
        else:
            print('data resize : %s -> %s'%(data.shape[:2], output_size))
            data=img_Resizer.img_resize(data)
            
        #temp, ta_temp,ir_min,ir_max
        print('convert temp 2 cwsi...')
        cwsi=temp_converter.convert_temp2cwsi(data,ta_temp,ir_min,ir_max)
        


        if target_path:
            print('calculate error...')
            try:
                file_name=[filename for filename in target_list if re.search(fn,filename)][0]
            except exception as e:
                print(e)
                continue
            target=pd.read_csv(target_path+file_name,header=None)
            error=calculator.mean_square_error(cwsi,target)
            print('%s mean square error :'%(fn), error)

        pd.DataFrame(cwsi).to_csv(result_path+fn+'.csv')
        
        if i > 1:
            break