from flask import Flask, request, jsonify
from flask_cors import CORS
import emotion_detection
from emotion_detection import ParallelModel


app = Flask(__name__)
CORS(app)

@app.route("/", methods = ['GET', 'POST'])
def response():
    if request.method == 'GET':
        return 'GET method is not available.'
    
    elif request.method =='POST':
        
        if request.is_json == False:
            return 'No json is sented.'
        
        params = request.get_json()
        wav_base64 = params['wav_base64']
        prd = emotion_detection.predict(wav_base64)
        
        data = {'emotion' : str(prd)}
        
        #angry, neutral, sadness, disgust, surprise, happiness, disgust
        return jsonify(data)

if __name__ == "__main__":
    
    
    # app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    app.run(debug=True, host="127.0.0.1", port=8080)
