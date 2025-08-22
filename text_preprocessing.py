"""
Text preprocessing functions for sentiment analysis
"""
import pandas as pd
import re

def clean_text(text):
    """
    Clean and preprocess text data for sentiment analysis
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text ready for vectorization
    """
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs and email addresses
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    return text

def create_sentiment_label(score):
    """
    Convert 1-5 star rating to sentiment category
    
    Args:
        score (int): Star rating (1-5)
        
    Returns:
        int: Sentiment label (0=Negative, 1=Neutral, 2=Positive)
    """
    if score <= 2:
        return 0  # Negative
    elif score == 3:
        return 1  # Neutral
    else:  # score >= 4
        return 2  # Positive

def validate_text_input(text):
    """
    Validate if input text is suitable for processing
    
    Args:
        text (str): Input text
        
    Returns:
        tuple: (is_valid, cleaned_text, message)
    """
    if not text or text.strip() == "":
        return False, "", "Text is empty"
    
    cleaned = clean_text(text)
    if len(cleaned) < 3:
        return False, cleaned, "Text too short after cleaning"
    
    return True, cleaned, "Text is valid"
