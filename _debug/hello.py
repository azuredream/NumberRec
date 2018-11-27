from flask import Flask ,redirect, url_for,render_template
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('jSig.html', message=1,picsrc = 1)

if __name__ == '__main__':
    app.run(debug = True)
