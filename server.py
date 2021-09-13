import random
import os
from flask import Flask, request, jsonify
from keyword_spotting_service import Keyword_Spotting_Service

"""
Server

client -> POST request -> server -> prediction back to client

"""


app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():

    # gets audio file and save it
    audio_file = request.files["file"]  # access to audio file
    file_name = str(random.randint(0, 100000))
    audio_file.save(file_name)  # file saved in working directory

    # invoke keyword spotting service
    kss = Keyword_Spotting_Service()

    # make prediction
    predicted_keyword = kss.predict(file_name)

    # remove audio file
    os.remove(file_name)

    # send predicted keyword to client
    data = {"keyword": predicted_keyword}

    return jsonify(data)

if __name__ == "__main__":
    app.run()       # debug=False