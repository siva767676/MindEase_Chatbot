import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

class MentalHealthPredictor:
    def __init__(self):
        self.model = None
        self.tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
        self.max_length = 100
        self.labels = ['Anxiety', 'Depression', 'Stress', 'Normal']
        
    def load_model(self):
        # Load the pre-trained model and tokenizer
        if os.path.exists('mental_health_model.h5') and os.path.exists('tokenizer.pickle'):
            self.model = tf.keras.models.load_model('mental_health_model.h5')
            with open('tokenizer.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            return True
        return False
    
    def predict(self, text):
        if self.model is None:
            if not self.load_model():
                return "Model not trained yet. Please train the model first."
        
        # Preprocess the input text
        sequences = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        # Make prediction
        prediction = self.model.predict(padded)
        predicted_label = self.labels[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        
        return {
            'label': predicted_label,
            'confidence': confidence,
            'probabilities': {
                label: float(prob) for label, prob in zip(self.labels, prediction[0])
            }
        }

# Initialize the predictor
predictor = MentalHealthPredictor() 