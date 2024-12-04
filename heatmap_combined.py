import pandas as pd
import gensim
from gensim import corpora
from gensim.models import LdaMulticore
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim.models.phrases import Phrases, Phraser
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path):
    print("Loading data from:", file_path)
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline()
            print("First line of the file:", first_line)
        data = pd.read_json(file_path, lines=True)
        data['timestamp'] = pd.to_datetime(data['created_utc'], unit='s')
        data['month'] = data['timestamp'].dt.to_period('M')
        return data
    except ValueError as e:
        print("Error loading JSON data:", e)
        return None

def preprocess_data(data):
    print("Starting data preprocessing...")
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    processed_texts = []
    for doc in data['body']:
        tokens = tokenizer.tokenize(doc.lower())
        processed_tokens = [token for token in tokens if token not in stop_words and len(token) > 3]
        processed_texts.append(processed_tokens)
    # Use Phrases to detect and form bigrams (two-word phrases)
    bigram = Phrases(processed_texts, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    processed_texts = [bigram_mod[doc] for doc in processed_texts]
    print("Data preprocessing completed.")
    return processed_texts

def build_topics(texts, num_topics=10):
    print(f"Building {num_topics} topics using LDA Multicore...")
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=10, workers=4)
    # Generate unique descriptions for each topic
    used_words = set()
    unique_topic_descriptions = {}
    for i in range(num_topics):
        words = [word for word, _ in lda_model.show_topic(i, topn=10) if word not in used_words]
        unique_topic_descriptions[i] = ', '.join(words[:5])
        used_words.update(words)
    return lda_model, corpus, dictionary, unique_topic_descriptions

def generate_heatmap(lda_model, corpus, data, topic_descriptions):
    print("Generating heatmap for topic distribution...")
    dominant_topics = [max(lda_model.get_document_topics(doc), key=lambda x: x[1])[0] for doc in corpus]
    data['dominant_topic'] = dominant_topics
    pivot_data = data.pivot_table(index='month', columns='dominant_topic', aggfunc='size', fill_value=0)
    pivot_data.columns = [topic_descriptions[i] for i in pivot_data.columns]
    sns.heatmap(pivot_data.T, cmap='viridis', annot=True)
    plt.title('Monthly Abortion Topics Heatmap')
    plt.xlabel('Months')
    plt.ylabel('Topics')
    plt.show()
    print("Heatmap displayed.")

def main(file_path):
    data = load_data(file_path)
    if data is not None:
        texts = preprocess_data(data)
        lda_model, corpus, dictionary, topic_descriptions = build_topics(texts)
        generate_heatmap(lda_model, corpus, data, topic_descriptions)

if __name__ == "__main__":
    file_path = '/Users/andresjiang/Desktop/RA/reddit/subreddits23/abortiondebates_comments'
    main(file_path)
