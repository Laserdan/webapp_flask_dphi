# Libraries
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os

# app name
app = Flask(__name__)

# load saved artifacts
def load_model():
    return pickle.load(open('artifacts/model.pkl', 'rb'))

def load_scaler():
    return pickle.load(open('artifacts/scaler.pkl', 'rb'))


# home page
@app.route('/')
def home():
    return render_template('index.html')


# predict the result and return it
@app.route('/predict', methods=['POST'])
def predict():
    features = [str(x).lower() for x in request.form.values()]
    gender = features[0]
    if gender == 'male':
        gender_lst = [0, 1]
    elif gender == 'female':
        gender_lst = [1, 0]

    married = features[1]
    if married == 'no':
        married_lst = [1, 0]
    elif married == 'yes':
        married_lst = [0, 1]

    dependents = features[2]
    if dependents == '0':
        dependents_lst = [1, 0, 0, 0]
    elif dependents == '1':
        dependents_lst = [0, 1, 0, 0]
    elif dependents == '2':
        dependents_lst = [0, 0, 1, 0]
    elif dependents == '3+':
        dependents_lst = [0, 0, 0, 1]

    education = features[3]
    if education == 'graduate':
        education_lst = [1, 0]
    elif education == 'not graduate':
        education_lst = [0, 1]

    selfemployed = features[4]
    if selfemployed == 'no':
        selfemployed_lst = [0, 1]
    elif selfemployed == 'yes':
        selfemployed_lst = [1, 0]

    credithistory = features[5]
    if credithistory == 'yes':
        credithistory_lst = [1]
    elif credithistory == 'no':
        credithistory_lst = [0]

    propertyarea = features[6]
    if propertyarea == 'rural':
        propertyarea_lst = [1, 0, 0]
    elif propertyarea == 'semiurban':
        propertyarea_lst = [0, 1, 0]
    elif propertyarea == 'urban':
        propertyarea_lst = [0, 0, 1]

    applicantincome = [int(features[7])]
    coapplicantincome = [int(features[8])]
    loanammount = [int(features[9])]
    loanammountterm = [int(features[10])]

    values_to_predict_raw = gender_lst + married_lst + dependents_lst + education_lst + selfemployed_lst + propertyarea_lst + applicantincome + coapplicantincome + loanammount + loanammountterm + credithistory_lst

    scaler = load_scaler()
    values_to_predict_scaled = scaler.transform(pd.DataFrame(values_to_predict_raw).T)

    model = load_model()
    prediction = model.predict(values_to_predict_scaled)

    labels = ['Loan Rejected', 'Loan accepted']
    result = labels[prediction[0]]

    return render_template('index.html', output='Status of the loan: {}'.format(result))


if __name__ == '__main__':
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)
