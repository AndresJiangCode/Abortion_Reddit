import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
import scispacy
from collections import Counter
import re

# 加载scispacy的医学模型
nlp = spacy.load("en_core_sci_lg")

def load_data(file_path):
    """加载数据"""
    return pd.read_json(file_path, lines=True)

def preprocess_text(text):
    """文本预处理和分词"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def extract_medical_terms(data):
    """提取医学术语和相关短语"""
    medical_terms = Counter()
    for doc in nlp.pipe(data['processed_text']):
        # 从scispacy模型获取医学实体
        ents = [ent.text for ent in doc.ents if ent.label_ in ['DISORDER', 'SYMPTOM']]
        medical_terms.update(ents)
    return medical_terms

def plot_heatmap(medical_terms):
    """生成热图显示医学术语的频率"""
    terms_df = pd.DataFrame(medical_terms.items(), columns=['Term', 'Count']).sort_values(by='Count', ascending=False).head(30)
    terms_df.set_index('Term', inplace=True)
    sns.heatmap(terms_df.T, annot=True, cmap='coolwarm', fmt='d')
    plt.title('Frequency of Negative Side Effects Related to Abortion')
    plt.xlabel('Terms')
    plt.ylabel('Frequency')
    plt.show()

def main(file_path):
    data = load_data(file_path)
    data['processed_text'] = data['body'].apply(preprocess_text)
    medical_terms = extract_medical_terms(data)
    plot_heatmap(medical_terms)

if __name__ == "__main__":
    file_path = '/Users/andresjiang/Desktop/RA/reddit/subreddits23/abortion_comments'
    main(file_path)