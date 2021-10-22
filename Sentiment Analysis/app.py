from flask import *


app=Flask(__name__)
app.secret_key="Hare Krishna"

@app.route('/')
def index():
    return render_template('index.')

if __name__=='main':
    app.run(debug=True)