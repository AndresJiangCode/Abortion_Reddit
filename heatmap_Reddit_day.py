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
        return data
    except ValueError as e:
        print(f"Error loading data: {e}")
        return None

def convert_timestamps(data):
    """将created_utc的时间戳转换为可读的日期"""
    data['date'] = pd.to_datetime(data['created_utc'], unit='s')
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    return data

def count_daily_comments(data):
    """计算每天的回复数量，包括年份和月份"""
    # 按照年、月和日分组并计数
    data['year_month'] = data['date'].dt.to_period('M')
    daily_counts = data.groupby(['year_month', 'day']).size().reset_index(name='counts')
    
    # 生成完整的月份范围
    min_month = data['date'].min().to_period('M')
    max_month = data['date'].max().to_period('M')
    all_months = pd.period_range(min_month, max_month, freq='M')
    
    # 生成完整的日期索引
    idx = pd.MultiIndex.from_product([all_months, range(1, 32)], names=['year_month', 'day'])
    daily_counts.set_index(['year_month', 'day'], inplace=True)
    daily_counts = daily_counts.reindex(idx, fill_value=0)  # 用0填充缺失的天数
    daily_counts = daily_counts.unstack(level=0).fillna(0)  # 转换为适合热图的形式

    return daily_counts

def plot_daily_counts(daily_counts):
    """绘制每天的回复数量热图，月份显示在垂直轴上，不显示具体计数"""
    plt.figure(figsize=(12, 10))  # 调整图形尺寸
    sns.heatmap(daily_counts.T, cmap='coolwarm', linewidths=.5)  # 使用 .T 来转置DataFrame
    plt.title('Daily Counts of Abortion Related Comments by Year and Month')
    plt.xlabel('Day of Month')
    plt.ylabel('Month, Year')  # 更新标签以反映变化
    plt.show()

def main(file_path):
    data = load_data(file_path)
    if data is None:
        return
    
    data = convert_timestamps(data)

    # Filter the data for dates from January 2022 onwards
    data = data[data['date'] >= '2022-01-01']

    daily_counts = count_daily_comments(data)
    plot_daily_counts(daily_counts)


if __name__ == "__main__":
    file_path = '/Users/andresjiang/Desktop/RA/reddit/subreddits23/abortion_submissions'
    main(file_path)