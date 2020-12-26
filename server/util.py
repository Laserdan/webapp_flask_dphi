import pickle
import pandas as pd
import numpy as np

__model = None
__scaler = None

def get_estimated_price(gender, married, dependents, education, selfemployed, applicantincome,
                        coapplicantincome, loanammount, loanammountterm, credithistory, propertyarea):

    #return __model.predict(array)
    pass

def load_saved_artifacts():
    print('loading saved artifacts...start')

    global  __model
    with open('./artifacts/model.pkl', 'rb') as f:
        __model = pickle.load(f)

    global __scaler
    with open('./artifacts/scaler.pkl', 'rb') as f:
        __scaler = pickle.load(f)
    print('loading saved artifacts...done')



if __name__ == '__main__':
    print('done')