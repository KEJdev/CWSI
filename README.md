## config 

* 실제 사용하기 위해 수정해야하는 파일은 **config.ini** 입니다.


모델이 저장된 경로는 아래와 같이 두가지 이며, 실제 Input이 온도면, Temp_model_Path에 있는 모델을 사용하며, Input이 RGB 데이터라면 rgb_model_Path에 있는 데이터를 사용합니다. 

```
temp_model_path = ./temp2cwsi/temp2cwsi_15000ckp.pkl
rgb_model_path = ./rgb2temp/rgb2temp_model.sav
```

Input 디렉토리는 아래와 같이 설정합니다. 

```
input_path = ./data/img_ir/
```

실제 학습할때 사용한 추가 데이터 ir_min, ir_max, ta_temp 저장된 디렉토리는 아래와 같습니다.

```
ta_path = ./data/csv_ta/metainfo.csv
```

모델 결과(예측) cwsi를 저장할 디렉토리는 아래와 같이 설정합니다. 

```
result_path = ./result/
```

실제 정답 데이터 저장 디렉토리는 아래와 같이 설정합니다. 

```
target_path = ./data/csv_cwsi/
```

## Model Parameter

만약 인풋 데이터가 온도면 Architecture='temp2cwsi' 로 설정하고 RGB 데이터면 Architecture='rgb2cwsi' 로 설정합니다. 

```
architecture = rgb2cwsi
```


모델 결과(예측) OutPut cwsi의 사이즈는 아래와 같이 설정 할 수 있습니다. 

```
output_size = (347, 435)
```

-------------

## Example   

**example.ipynb**에서 테스트 할 수 있습니다.  

1. RGB 데이터 -> CWSI 변환

```python
# 예측 모델을 불러옵니다. 
from main import predict_cwsi,predict_cwsi2

import configparser
config = configparser.ConfigParser()
config.read('config.ini',encoding='utf-8') # 위에서 설정했던 디렉토리 경로들을 불러옵니다.

# rgb -> cwsi 
predict_cwsi(config) 
```

2. 온도 데이터 -> CWSI 변환

```python 
from main import predict_cwsi,predict_cwsi2

# temp -> cwsi
config['path']['input_path']='./data/img_temp_resize/'
config['model_parameter']['architecture']='temp2cwsi'

predict_cwsi(config)
```

-------------

※ 주의사항으로 각 폴더 rgb2temp와 temp2cwsi안에 있는 data 파일은 지우시면 안됩니다. 


