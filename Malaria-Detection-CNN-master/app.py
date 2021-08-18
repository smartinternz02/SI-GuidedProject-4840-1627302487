import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask , request, render_template



app = Flask(__name__)
model = load_model("malaria_detector.h5")

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")

        basepath = os.path.dirname(__file__)
        print("current path", basepath)

        filepath = os.path.join(basepath,'static/pics',f.filename)
        print("upload folder is ", filepath)

        f.save(filepath)

        file = "/static/pics/" + f.filename


        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        pred = model.predict(x)

        print("prediction",pred)

        list = ["Parasitized", "Uninfected"]

        print(np.argmax(pred))

        result = "Result : " + str(list[np.argmax(pred)])

    return render_template("index.html", result=result, uploaded_image=file)

if __name__ == '__main__':
    app.run(debug = True)
