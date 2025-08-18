# Amazon Reviews Sentiment Analysis Model

A complete machine learning system for analyzing customer sentiment in Amazon product reviews.

## Model Overview

- **Model Type**: Logistic Regression
- **Feature Extraction**: TF-IDF  
- **Accuracy**: 0.7337 (73.37%)
- **Classes**: Negative, Neutral, Positive
- **Training Date**: 2025-08-17 02:25:11
- **Dataset Size**: 127,920 balanced reviews

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from predict_sentiment import SentimentAnalyzer

# Initialize the analyzer
analyzer = SentimentAnalyzer()

# Analyze a single review
result = analyzer.predict_single("This product is amazing!")
print(f"Sentiment: {result['predicted_sentiment']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Batch Processing
```python
reviews = [
    "Love this product!",
    "Terrible quality", 
    "It's okay"
]
results = analyzer.predict_batch(reviews)
```

## Files Description

| File | Description |
|------|-------------|
| `best_model.pkl` | Trained Logistic Regression model |
| `vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `predict_sentiment.py` | Main prediction script with SentimentAnalyzer class |
| `text_preprocessing.py` | Text cleaning and validation functions |
| `model_info.json` | Model metadata and performance metrics |
| `label_mapping.json` | Sentiment label mappings and descriptions |
| `requirements.txt` | Python package dependencies |

## Model Performance

### Training Results
- **Training Samples**: 102,336
- **Testing Samples**: 25,584
- **Training Time**: 4.38 seconds
- **Vocabulary Size**: 10,000 features

### Comparison Results
- **Logistic Regression (Best)**: 0.7337
- **Naive Bayes (Best)**: 0.6890
- **TF-IDF vs Count**: TF-IDF performed better

## Technical Details

### Sentiment Classes
- **Negative (0)**: 1-2 star reviews (dissatisfied customers)
- **Neutral (1)**: 3 star reviews (mixed feelings)
- **Positive (2)**: 4-5 star reviews (satisfied customers)

### Text Preprocessing
1. Convert to lowercase
2. Remove HTML tags and URLs
3. Remove special characters (keep only letters)
4. Remove extra whitespaces
5. Validate minimum length

### Feature Engineering
- **Method**: TF-IDF
- **Max Features**: 10,000
- **N-grams**: 1-2 (unigrams + bigrams)
- **Min Document Frequency**: 5
- **Max Document Frequency**: 90%

## Usage Examples

### Command Line Testing
```bash
python predict_sentiment.py
```

### Integration Example
```python
import joblib
from text_preprocessing import clean_text

# Load components manually
model = joblib.load('best_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Process new review
def analyze_review(text):
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0].max()
    
    labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return labels[prediction], confidence

# Example usage
sentiment, conf = analyze_review("Great product, highly recommended!")
print(f"{sentiment} ({conf:.3f})")
```

## Model Validation

The model was tested on a balanced dataset with equal representation of all sentiment classes. Key validation metrics:

- **Cross-validation**: Stratified train-test split (80-20)
- **Balance**: 42,640 samples per class
- **Preprocessing**: Comprehensive text cleaning pipeline
- **Feature Selection**: Optimized TF-IDF/Count vectorization
- **Model Comparison**: Multiple algorithms tested

## Business Applications

- **Customer Feedback Analysis**: Automatically categorize customer reviews
- **Product Quality Monitoring**: Track sentiment trends over time  
- **Marketing Insights**: Understand customer satisfaction patterns
- **Support Prioritization**: Identify dissatisfied customers quickly
- **Competitive Analysis**: Compare sentiment across products

## Customization

### Retraining the Model
```python
# Load your new data
new_df = pd.read_csv('new_reviews.csv')

# Use the same preprocessing pipeline
from text_preprocessing import clean_text, create_sentiment_label

# Apply preprocessing and retrain
# (Follow the same steps as in the original notebook)
```

### Modifying Classes
To use binary classification (Positive/Negative only):
```python
# Modify the create_sentiment_label function
def binary_sentiment(score):
    return 0 if score <= 3 else 1  # Negative vs Positive
```

## Limitations & Considerations

- **Domain Specific**: Trained on Amazon product reviews
- **English Only**: Designed for English text
- **Context**: May not capture sarcasm or complex contexts
- **Retraining**: Recommend periodic retraining with fresh data
- **Preprocessing**: Input text should follow same cleaning process

## Support & Maintenance

- **Model Version**: v1.0
- **Created**: 2025-08-17
- **Next Review**: Recommended after 6 months
- **Performance Monitoring**: Track accuracy on new data

---

**Ready to analyze customer sentiment at scale!**

For questions or improvements, please refer to the original training notebook for detailed methodology and implementation details.
