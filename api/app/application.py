from flask import Flask,jsonify,request,session
from flask_session import Session
from utils import preprocess,describe

prepro = preprocess()
desc = describe()

app = Flask(__name__)

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route('/')
def index():
   return 'Hello ! Welcome to Dr.Crop API system'

@app.route('/soil',methods=['POST'])
def soil():
   data=request.get_json()
   n = data['n']
   p = data['p']
   k = data['k']
   temp = data['temp']
   humidity = data['humidity']
   ph = data['pH']
   rainfall = data['rainfall']
   data_final = [n,p,k,temp,humidity,ph,rainfall]
   prediction = prepro.predict_crop(data_final)
   data_final.append(prediction)
   text = desc.crop(data_final)
   return jsonify(
      {"crop" : prediction,
      "Suggestion" : text})

@app.route('/leaf', methods=['POST'])
def leaf():
   data = request.get_json()

   prediction,accuracy = prepro.predict_disease(data)
   return jsonify(
      {'crop':data['crop'],
      'disease':prediction,
      'accuracy':str(accuracy*100)}
   )

@app.route('/soil/preferance',methods=['POST'])
def preferance():
   data = request.get_json()
   n = data['n']
   p = data['p']
   k = data['k']
   temp = data['temp']
   humidity = data['humidity']
   ph = data['pH']
   rainfall = data['rainfall']
   crop = data['crop']
   data_final = [n,p,k,temp,humidity,ph,rainfall,crop]
   suggestion = desc.preferance(data_final)
   return jsonify({
      'suggestion':suggestion
   })
if __name__ == "__main__":
    app.run()