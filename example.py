from main import predict_cwsi
import configparser
config = configparser.ConfigParser()
config.read('config.ini',encoding='utf-8')

if __name__=='__main__':
    predict_cwsi(config)