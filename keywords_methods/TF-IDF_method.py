import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict

# Ensure the necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def load_data(file_path):
    """Loads data from a JSON file."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    try:
        data = pd.read_json(file_path, lines=True)
        return data
    except ValueError as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(text):
    """Preprocesses text data by cleaning and converting to lowercase."""
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\S*\.com\S*', '', text)  # Remove .com and related parts
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove words that are <= 2 characters
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    stopped_tokens = [i for i in tokens if not i in stopwords.words('english')]
    return stopped_tokens

def calculate_cooccurrence(data, top_keywords, window_size=4):
    """Calculates cooccurrence of top keywords with other words in the text."""
    cooccurrence = defaultdict(int)
    for document in data:
        for i, word in enumerate(document):
            if word in top_keywords:
                start = max(i - window_size, 0)
                end = min(i + window_size + 1, len(document))
                for j in range(start, end):
                    if i != j:
                        cooccurrence[(word, document[j])] += 1
    return cooccurrence

def main(file_path):
    data = load_data(file_path)
    if data is None:
        return

    # Preprocess data and join tokens back into single strings per document
    processed_data = data['body'].dropna().apply(preprocess_data)
    processed_data = [' '.join(doc) for doc in processed_data if doc]  # Rejoin words into single strings

    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    
    # Apply TF-IDF transformation
    try:
        tfidf_matrix = vectorizer.fit_transform(processed_data)
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract the top TF-IDF keywords
        sorted_items = sorted(zip(vectorizer.idf_, feature_names))
        top_keywords = [word for score, word in sorted_items[:20]]
        
        print("Top TF-IDF Keywords:", top_keywords)
    except Exception as e:
        print(f"Error processing TF-IDF: {e}")

if __name__ == "__main__":
    file_path = '/Users/andresjiang/Desktop/RA/reddit/subreddits23/abortiondebate_comments'
    main(file_path)