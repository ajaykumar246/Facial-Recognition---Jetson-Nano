from flask import Flask
from test_face_recognition import face_check
app = Flask(__name__)

@app.route("/predict")
def predict():
	ans = face_check()
	return ans

if __name__ == "__main__":
	app.run(host="0.0.0.0",port=4123,debug=True)