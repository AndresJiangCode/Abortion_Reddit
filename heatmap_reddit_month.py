import pandas as pd
import seaborn as sns
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

def preprocess_data(data):
    """数据预处理：提取月份和计数"""
    monthly_counts = data.groupby('month').size().reset_index(name='counts')
    return monthly_counts

def plot_heatmap(monthly_counts):
    """绘制热图，纵轴为月份，横轴为计数"""
    monthly_counts_pivot = monthly_counts.pivot("month", "counts", "counts")
    plt.figure(figsize=(12, 8))
    sns.heatmap(monthly_counts_pivot, cmap='coolwarm', annot=False)  # 关闭注释以清除计数标签
    plt.title('Monthly Counts of Abortion Related Comments/Submissions from 2021')
    plt.xlabel('Counts')
    plt.ylabel('Month')
    plt.show()

def main(file_path):
    data = load_data(file_path)
    if data is None:
        return
    data = data[data['timestamp'] >= '2021-01-01']  # Filter for data from 2021 onwards
    monthly_counts = preprocess_data(data)
    plot_heatmap(monthly_counts)


if __name__ == "__main__":
    file_path = '/Users/andresjiang/Desktop/RA/reddit/subreddits23/abortion_submissions'
    main(file_path)