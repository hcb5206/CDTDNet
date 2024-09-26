import pandas as pd

# 从CSV文件中读取原始数据
input_file = "C:\\Users\\HE CONG BING\\Desktop\\station_1001.csv"
data = pd.read_csv(input_file)

# 将日期时间列转换为 datetime 格式
data['time'] = pd.to_datetime(data['time'])

# 定义起始和结束时间
start_time = pd.to_datetime("2014-05-01 00:00:00")
end_time = pd.to_datetime("2015-04-30 22:00:00")
time_step = pd.DateOffset(hours=1)

# 创建一个时间戳的列表
timestamps = []
current_time = start_time
while current_time <= end_time:
    timestamps.append(current_time)
    current_time += time_step

# 创建一个新的DataFrame，以包含所有的时间戳
new_data = pd.DataFrame(index=timestamps)

# 使用 left join 将新的DataFrame与原数据合并，将缺失值设置为 NaN
merged_data = new_data.merge(data, left_index=True, right_on='time', how='left')

# 保存处理后的数据到新的CSV文件
output_file = "C:\\Users\\HE CONG BING\\Desktop\\station_1001_1.csv"
merged_data.to_csv(output_file, index=False)  # 不保存索引
