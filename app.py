from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('Classification_chiffres_manuscrits.keras')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    image = Image.open(image_file).convert('L')
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return jsonify({'prediction': int(predicted_class)})


if __name__ == '__main__':
    app.run(debug=True)
