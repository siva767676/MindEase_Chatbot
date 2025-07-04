from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import google.generativeai as genai
import os
from dotenv import load_dotenv
from chatbot import Chatbot

# Load environment variables+
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)

# Initialize the chat model
model = genai.GenerativeModel("models/gemini-1.5-flash")
chat = model.start_chat(history=[])

# Initialize chatbot
chatbot = Chatbot()

# Initialize tokenizer and label mapping with default values
tokenizer = None
label_mapping = None

# Load tokenizer and label mappings
try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_mapping.pickle', 'rb') as handle:
        label_mapping = pickle.load(handle)
except Exception as e:
    print(f"Error loading tokenizer or label mapping: {e}")
    # Create a default tokenizer if loading fails
    tokenizer = Tokenizer(num_words=10000)
    # Default label mapping
    label_mapping = {0: 'normal', 1: 'depression', 2: 'anxiety', 3: 'stress'}

# Custom model loading with error handling
try:
    # Try loading with custom_objects to handle batch_shape
    mental_health_model = tf.keras.models.load_model('mental_health_model.h5', 
        custom_objects={'batch_shape': None})
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback: Try recreating the model architecture
    mental_health_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(10000, 128),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    # Load weights only
    try:
        mental_health_model.load_weights('mental_health_model.h5')
    except Exception as e:
        print(f"Error loading weights: {e}")
        # Compile the model even if weights loading fails
        mental_health_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        user_message = request.json['message']
        response = chat.send_message(user_message)
        return jsonify({'response': response.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.json['text']
        
        # Fit tokenizer on the input text if it's new
        if not tokenizer.word_index:
            tokenizer.fit_on_texts([text])
        
        # Preprocess the input text
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')
        
        # Make prediction
        prediction = mental_health_model.predict(padded_sequence, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        label = label_mapping[predicted_class]
        
        # Get probabilities for all classes
        probabilities = {
            label_mapping[i]: float(prediction[0][i])
            for i in range(len(label_mapping))
        }
        
        return jsonify({
            'success': True,
            'prediction': label,
            'confidence': confidence,
            'probabilities': probabilities
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True) 