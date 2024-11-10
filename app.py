from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Charger le modèle
model = tf.keras.models.load_model('mon_modele.keras')

# Route principale pour afficher l'interface de téléchargement d'image
@app.route('/')
def home():
    # HTML pour l'interface d'upload
    html = """
    <!doctype html>
    <title>Prédiction de chiffre manuscrit</title>
    <h1>Uploader une image pour prédire le chiffre manuscrit</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*">
        <input type="submit" value="Prédire">
    </form>
    {% if prediction is not none %}
        <h2>Prédiction: {{ prediction }}</h2>
    {% endif %}
    """
    return render_template_string(html, prediction=None)

# Route pour gérer la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "Aucune image détectée", 400

    image_file = request.files['image']
    image = Image.open(image_file).convert('L')  # Convertir en niveaux de gris
    image = image.resize((28, 28))  # Redimensionner
    image = np.array(image) / 255.0  # Normaliser
    image = np.expand_dims(image, axis=0)  # Ajouter la dimension pour le batch

    # Prédire avec le modèle
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Retourner la page avec le résultat
    html = """
    <!doctype html>
    <title>Prédiction de chiffre manuscrit</title>
    <h1>Uploader une image pour prédire le chiffre manuscrit</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*">
        <input type="submit" value="Prédire">
    </form>
    <h2>Prédiction: {{ prediction }}</h2>
    """
    return render_template_string(html, prediction=predicted_class)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
