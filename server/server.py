from flask import Flask, request, jsonify
import util

app = Flask(__name__, static_url_path="/client", static_folder='../client', template_folder="../client")

@app.route('/predict_loan', methods=['POST'])
def predict_loan():
    gender = request.form['gender']
    married = request.form['married']
    dependents = request.form['dependents']
    education = request.form['education']
    selfemployed = request.form['selfemployed']
    applicantincome = int(request.form['applicantincome'])
    coapplicantincome = int(request.form['coapplicantincome'])
    loanammount = int(request.form['loanammount'])
    loanammountterm = int(request.form['loanammountterm'])
    credithistory = request.form['credithistory']
    propertyarea = request.form['propertyarea']

    response = jsonify({
        'prediction': util.get_estimated_price(gender, married, dependents, education, selfemployed,
                                               applicantincome, coapplicantincome, loanammount,
                                               loanammountterm, credithistory, propertyarea)
    })

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == "__main__":
    print('Starting Python Flask server for loan or not loan prediction...')
    util.load_saved_artifacts()
    app.run(debug=True)
