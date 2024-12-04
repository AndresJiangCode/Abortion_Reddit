import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

def load_data(directory_path):
    """加载指定文件夹中所有.txt文件并合并为一个DataFrame"""
    # 使用glob来找到所有的文件
    files = glob.glob(os.path.join(directory_path, '*'))
    if not files:
        print("No . files found in the directory.")
        return None
    data_frames = []
    for file in files:
        if not os.path.exists(file):
            print(f"File not found: {file}")
            continue
        try:
            data = pd.read_json(file, lines=True)
            data_frames.append(data)
        except ValueError as e:
            print(f"Error loading data from {file}: {e}")
    
    if not data_frames:
        print("No data loaded.")
        return None
    # 合并所有DataFrame
    return pd.concat(data_frames, ignore_index=True)

def convert_timestamps(data):
    """将created_utc的时间戳转换为可读的日期"""
    data['date'] = pd.to_datetime(data['created_utc'], unit='s')
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    return data

def count_daily_comments(data):
    """计算每天的回复数量，包括年份和月份"""
    data['year_month'] = data['date'].dt.to_period('M')
    daily_counts = data.groupby(['year_month', 'day']).size().reset_index(name='counts')
    # 创建完整的月份日历
    min_month = data['date'].min().to_period('M')
    max_month = data['date'].max().to_period('M')
    idx = pd.period_range(min_month, max_month, freq='M')
    daily_counts_pivot = daily_counts.pivot('year_month', 'day', 'counts').reindex(idx, fill_value=0)
    return daily_counts_pivot

def plot_daily_counts(daily_counts):
    """绘制每天的回复数量热图"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(daily_counts, cmap='coolwarm', linewidths=.5)
    plt.title('Daily Counts of Abortion Related Comments by Year and Month')
    plt.xlabel('Day of Month')
    plt.ylabel('Year and Month')
    plt.show()

def main(directory_path):
    data = load_data(directory_path)
    if data is None:
        return
    data = convert_timestamps(data)
    daily_counts = count_daily_comments(data)
    plot_daily_counts(daily_counts)

if __name__ == "__main__":
    directory_path = '/Users/andresjiang/Desktop/RA/reddit/subreddits23/comments'
    main(directory_path)