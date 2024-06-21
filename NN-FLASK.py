from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = load_model('model/combined_model.h5')

# FER emotion scores mapping
fer_emotion_scores = {
    'angry': 0.2,
    'disgust': 0.3,
    'fear': 0.4,
    'happy': 1.0,
    'neutral': 0.5,
    'sad': 0.1,
    'surprise': 0.8
}

# Initialize the LabelEncoder and fit it with the final labels
final_labels = ['stressed', 'slightly stressed', 'neutral', 'okay', 'happy']
label_encoder = LabelEncoder()
label_encoder.fit(final_labels)

# Function to get FER score
def get_fer_score(fer_label):
    return fer_emotion_scores[fer_label]

# Weighted average function
def weighted_average(sentiment_score, fer_score, sentiment_weight=0.6, fer_weight=0.4):
    combined_score = (sentiment_score * sentiment_weight) + (fer_score * fer_weight)
    return combined_score

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.json
    sentiment_score = data.get('sentiment_score')
    fer_label = data.get('fer_label')

    # Check if inputs are valid
    if sentiment_score is None or fer_label is None or fer_label not in fer_emotion_scores:
        return jsonify({'error': 'Invalid input'}), 400

    # Get FER score
    fer_score = get_fer_score(fer_label)

    # Calculate combined score
    combined_score = weighted_average(sentiment_score, fer_score)

    # Predict using the model
    combined_score = np.array([[combined_score]])
    prediction = model.predict(combined_score)
    predicted_class = tf.argmax(prediction, axis=1).numpy()[0]

    # Decode the predicted class to the final label
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    # Return the prediction
    return jsonify({'predicted_label': predicted_label})

# Run the Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)
