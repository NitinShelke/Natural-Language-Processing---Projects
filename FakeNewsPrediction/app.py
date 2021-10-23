from flask import Flask,render_template,request
from com_in_nitin_predict.predictAPI import predict
from logging import FileHandler,WARNING,ERROR

app = Flask(__name__)

file_handler = FileHandler('errorlog.txt')
file_handler.setLevel(ERROR)


@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict1',methods=['POST'])
def predict1():
    news=request.form['news']
    prediction=predict(news)

    if prediction[0]==0:
        return render_template('index.html',prediction_text='Great It is a real news!')
    else:
        return  render_template('index.html',prediction_text='Careful: seems fake news!')


if __name__ == '__main__':
    app.run(debug=True)
