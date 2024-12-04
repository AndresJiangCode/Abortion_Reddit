import pandas as pd
import gensim
from gensim import corpora, models
from gensim.models import LdaMulticore
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path):
    print("Loading data...")
    return pd.read_json(file_path, lines=True)

def preprocess_data(data):
    print("Starting data preprocessing...")
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    print(f"Tokenizing and removing stopwords from {len(data)} documents...")
    texts = data['body'].map(lambda x: tokenizer.tokenize(x.lower()))
    texts = texts.map(lambda x: [word for word in x if word not in stop_words and len(word) > 3])
    print("Data preprocessing completed.")
    return texts

def build_topics(texts, num_topics=20):
    print(f"Building {num_topics} topics using LDA Multicore...")
    dictionary = corpora.Dictionary(texts)
    print("Dictionary created. Filtering extremes...")
    dictionary.filter_extremes(no_below=15, no_above=0.5)
    print(f"Dictionary size after filtering: {len(dictionary.keys())} tokens")
    corpus = [dictionary.doc2bow(text) for text in texts]
    print("Corpus created. Starting LDA model training...")
    lda_model = LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=10, workers=4)
    topics = lda_model.show_topics(formatted=False, num_words=10)
    topic_descriptions = [" ".join([word for word, prob in topic]) for topic_num, topic in topics]
    print("LDA model training completed.")
    return lda_model, corpus, dictionary, topic_descriptions

def generate_heatmap(lda_model, corpus, topic_descriptions):
    print("Generating heatmap for topic distribution...")
    num_topics = len(topic_descriptions)  # Ensure the size is consistent with the number of topics
    topic_prevalence = np.zeros(num_topics)  # Initialize with the correct size

    # Accumulate topic prevalence
    for doc in corpus:
        for topic_num, prop_topic in lda_model.get_document_topics(doc, minimum_probability=0):
            if topic_num < num_topics:  # Ensure the topic number is within the expected range
                topic_prevalence[topic_num] += prop_topic

    # Normalize the topic prevalence by the number of documents to get an average prevalence per topic
    topic_prevalence /= len(corpus)

    # Plotting
    sns.heatmap(topic_prevalence[:, np.newaxis], annot=True, fmt=".2f", cmap='viridis', yticklabels=topic_descriptions)
    plt.title('Top 20 Abortion Topics Heatmap')
    plt.xlabel('Topic Prevalence')
    plt.ylabel('Topics')
    plt.show()
    print("Heatmap displayed.")

def main(file_path):
    data = load_data(file_path)
    texts = preprocess_data(data)
    lda_model, corpus, dictionary, topic_descriptions = build_topics(texts, 20)  # Ensure you pass 20 or adjust as needed
    generate_heatmap(lda_model, corpus, topic_descriptions)

if __name__ == "__main__":
    file_path = '/Users/andresjiang/Desktop/RA/reddit/subreddits23/abortiondebates_comments'
    main(file_path)