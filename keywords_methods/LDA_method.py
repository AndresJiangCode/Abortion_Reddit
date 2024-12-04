import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models
from collections import defaultdict,Counter

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

def prepare_text_for_lda(data):
    """Prepares text data for LDA analysis."""
    if 'body' not in data.columns:
        return [], None
    text_data = []
    data['body'].dropna().apply(lambda x: text_data.append(preprocess_data(x)))
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    return text_data, dictionary, corpus

def apply_lda(corpus, dictionary, num_topics=5):
    """Applies LDA to the given corpus and dictionary."""
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    topics = lda_model.print_topics(num_words=10)
    return topics, lda_model

def extract_keywords_from_file(file_path):
    """Extracts topics related to 'abortion' from the file."""
    data = load_data(file_path)
    if data is None:
        return [], None, None
    text_data, dictionary, corpus = prepare_text_for_lda(data)
    if not text_data:
        return [], None, None
    topics, lda_model = apply_lda(corpus, dictionary, num_topics=5)
    return topics, lda_model, corpus, dictionary

file_path = '/Users/andresjiang/Desktop/RA/reddit/subreddits23/abortiondebates_comments'
topics, lda_model, corpus, dictionary = extract_keywords_from_file(file_path)
print("LDA Topics:", topics)

# To further analyze keywords for each topic
def further_analyze_keywords(lda_model, corpus, dictionary):
    topic_terms = {i: [dictionary[word_id] for word_id, value in lda_model.get_topic_terms(i)] for i in range(lda_model.num_topics)}
    word_relations = defaultdict(list)
    for topic_id, words in topic_terms.items():
        for doc in corpus:
            doc_words = [dictionary[word_id] for word_id, freq in doc if dictionary[word_id] in words]
            for word in doc_words:
                word_relations[word].extend([dictionary[word_id] for word_id, freq in doc if dictionary[word_id] != word])
    related_keywords = {word: dict(Counter(rel_words)) for word, rel_words in word_relations.items()}
    return related_keywords

related_keywords = further_analyze_keywords(lda_model, corpus, dictionary)
print("Related Keywords:", related_keywords)