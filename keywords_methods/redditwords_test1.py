import pandas as pd
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')


def load_data(file_path):
    try:
        data = pd.read_json(file_path, lines=True)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(text):
    """预处理文本数据：清洗和转换为小写"""
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lower case
    return text

def tokenize(text):
    """分词处理"""
    return word_tokenize(text)

def remove_stopwords(words):
    """去除停用词"""
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words

def count_word_frequencies(words):
    """计算词频"""
    return Counter(words)

def analyze_relevance(word_frequencies, target_word="abortion"):
    """分析与目标词相关的词汇"""
    # Assuming 'abortion' must be in the list, else return empty list
    if target_word not in word_frequencies:
        return []
    # Simply return words that co-occur highly with the target word
    return word_frequencies.most_common(20)

def extract_keywords_from_file(file_path):
    """从文件中提取与 'abortion' 相关的关键词"""
    data = load_data(file_path)
    if data is None:
        return []

    # Assuming there is a column named 'body' with relevant contents
    data['cleaned_text'] = data['title'].apply(preprocess_data)
    
    all_words = []
    data['cleaned_text'].dropna().apply(lambda x: all_words.extend(tokenize(x)))
    
    filtered_words = remove_stopwords(all_words)
    word_frequencies = count_word_frequencies(filtered_words)
    
    related_keywords = analyze_relevance(word_frequencies, "abortion")
    
    return related_keywords

file_path = '/Users/andresjiang/Desktop/RA/reddit/subreddits23/abortion_submissions'
keywords = extract_keywords_from_file(file_path)
print("Related Keywords:", keywords)
