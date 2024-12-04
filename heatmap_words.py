import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(file_path):
    """加载数据从JSON文件"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    try:
        data = pd.read_json(file_path, lines=True)
        data['timestamp'] = pd.to_datetime(data['created_utc'], unit='s')
        data['month'] = data['timestamp'].dt.to_period('M')  # 转换为年-月格式
        return data
    except ValueError as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data, keywords):
    """数据预处理：提取包含特定关键词的条目，并按月份计数"""
    # 过滤含有关键词的记录
    filtered_data = data[data['body'].str.contains('|'.join(keywords), case=False, na=False)]
    # 按月计算每个关键词的出现次数
    monthly_keyword_counts = filtered_data.groupby('month')['body'].apply(lambda x: x.str.contains('|'.join(keywords)).sum()).reset_index(name='counts')
    return monthly_keyword_counts

def plot_heatmap(data, title):
    """绘制热图，纵轴为月份，横轴为计数"""
    plt.figure(figsize=(10, 6))
    pivot_table = data.pivot_table(index='month', values='counts', aggfunc='sum')
    plt.imshow(pivot_table, cmap='coolwarm', aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Counts')
    plt.ylabel('Month')
    plt.xticks(range(len(pivot_table.columns)), pivot_table.columns, rotation=90)
    plt.yticks(range(len(pivot_table.index)), [str(x) for x in pivot_table.index])
    plt.show()

def main(file_path):
    keywords = ['depression', 'anxiety']
    data = load_data(file_path)
    if data is None:
        return
    data = data[data['timestamp'] >= '2021-01-01']  # Filter for data from 2021 onwards
    monthly_counts = preprocess_data(data, keywords)
    plot_heatmap(monthly_counts, 'Monthly Frequency of Keywords "Depression" and "Anxiety" from 2021')

if __name__ == "__main__":
    file_path = '/Users/andresjiang/Desktop/RA/reddit/subreddits23/abortion_submissions'
    main(file_path)
