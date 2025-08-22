"""
Sentiment Analysis Prediction Script
Load trained model and make predictions on new text data
"""
import joblib
import json
import numpy as np
from text_preprocessing import clean_text, validate_text_input

class SentimentAnalyzer:
    """
    Complete sentiment analysis pipeline
    """
    
    def __init__(self, model_folder="."):
        """Initialize the sentiment analyzer by loading saved components"""
        self.model_folder = model_folder
        self.model = None
        self.vectorizer = None
        self.label_mapping = None
        self.metadata = None
        self._load_components()
    
    def _load_components(self):
        """Load all saved model components"""
        try:
            # Load trained model
            self.model = joblib.load(f"{self.model_folder}/best_model.pkl")
            
            # Load vectorizer
            self.vectorizer = joblib.load(f"{self.model_folder}/vectorizer.pkl")
            
            # Load label mapping
            with open(f"{self.model_folder}/label_mapping.json", 'r') as f:
                self.label_mapping = json.load(f)
            
            # Load metadata
            with open(f"{self.model_folder}/model_info.json", 'r') as f:
                self.metadata = json.load(f)
                
            print(f"Model loaded successfully!")
            print(f"Model: {self.metadata['model_info']['model_type']}")
            print(f"Accuracy: {self.metadata['model_info']['accuracy']:.4f}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def predict_single(self, text):
        """
        Predict sentiment for a single text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Prediction results with sentiment, confidence, and probabilities
        """
        # Validate input
        is_valid, cleaned_text, message = validate_text_input(text)
        if not is_valid:
            return {'error': message, 'text': text}
        
        try:
            # Transform text to features
            features = self.vectorizer.transform([cleaned_text])
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            # Get sentiment name
            sentiment_name = self.label_mapping['labels_to_names'][str(prediction)]
            confidence = float(probabilities.max())
            
            # Create probability dict
            prob_dict = {
                name: float(probabilities[label]) 
                for label, name in self.label_mapping['labels_to_names'].items()
            }
            
            return {
                'text': text[:100] + "..." if len(text) > 100 else text,
                'cleaned_text': cleaned_text,
                'predicted_sentiment': sentiment_name,
                'confidence': confidence,
                'probabilities': prob_dict,
                'prediction_label': int(prediction)
            }
            
        except Exception as e:
            return {'error': str(e), 'text': text}
    
    def predict_batch(self, texts):
        """
        Predict sentiment for multiple texts
        
        Args:
            texts (list): List of texts to analyze
            
        Returns:
            list: List of prediction results
        """
        results = []
        for i, text in enumerate(texts):
            print(f"Processing {i+1}/{len(texts)}...")
            result = self.predict_single(text)
            results.append(result)
        
        return results
    
    def get_model_info(self):
        """Return model information and performance metrics"""
        return self.metadata

# Example usage and testing
if __name__ == "__main__":
    print("Testing Sentiment Analyzer...")
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Test examples
    test_reviews = [
        "This product is absolutely amazing! I love it so much!",
        "Terrible quality, complete waste of money. Very disappointed.",
        "It's okay, nothing special but does the job fine.",
        "Best purchase ever! Highly recommend to everyone!",
        "Cheap material, broke after one day. Avoid this!",
        "Good value for money, works as expected.",
    ]
    
    print(f"\nTesting with {len(test_reviews)} sample reviews:")
    print("=" * 80)
    
    for i, review in enumerate(test_reviews, 1):
        result = analyzer.predict_single(review)
        
        if 'error' in result:
            print(f"\nTest {i}: Error - {result['error']}")
        else:
            sentiment = result['predicted_sentiment']
            confidence = result['confidence']
            
            # Color coding for terminal output
            color = "green" if sentiment == "Positive" else "red" if sentiment == "Negative" else "yellow"
            
            print(f"\nTest {i}: {color} {sentiment} ({confidence:.3f})")
            print(f"Review: {result['text']}")
            print(f"Probabilities: Neg={result['probabilities']['Negative']:.3f}, "
                  f"Neu={result['probabilities']['Neutral']:.3f}, "
                  f"Pos={result['probabilities']['Positive']:.3f}")
    
    print(f"\nTesting completed successfully!")
    print(f"Model ready for production use!")
