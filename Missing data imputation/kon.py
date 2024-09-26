import pandas as pd
import numpy as np
import statsmodels.api as sm

# 从CSV文件中读取数据
input_file = "C:\\Users\\HE CONG BING\\Desktop\\station_1001_1.csv"
data = pd.read_csv(input_file)

# 假设data包含时间戳列"time"和需要填补的特征列
# 特征列的列名
feature_columns = [
    "PM25_Concentration", "PM10_Concentration", "NO2_Concentration", "CO_Concentration", "O3_Concentration",
    "SO2_Concentration"]

# 将时间戳列转换为日期时间格式
data['time'] = pd.to_datetime(data['time'])

# 复制原始数据以防止修改原始数据
filled_data = data.copy()

np.set_printoptions(precision=1)


# 定义一个函数来填补空缺值
def fill_missing_values(df, feature_column):
    # 获取缺失值的索引
    missing_indices = df[df[feature_column].isnull()].index
    print(len(missing_indices))

    i = 1
    # 遍历每个缺失值的索引
    for idx in missing_indices:
        # 获取当前时间戳
        current_time = df.loc[idx, 'time']

        # 找到之前的时间戳，也就是空缺值之前的数据
        previous_data = df[df['time'] < current_time]
        # print(previous_data)

        if not previous_data.empty:
            # 训练ARIMA模型
            p, d, q = 1, 1, 1  # ARIMA模型的阶数，可以根据需要进行调整
            # print(previous_data[feature_column])
            model = sm.tsa.ARIMA(previous_data[feature_column], order=(p, d, q))
            model_fit = model.fit()

            # 预测并填补缺失值
            predicted_value = abs(model_fit.forecast(steps=1).to_numpy())
            print(i, feature_column, predicted_value)
            i += 1
            df.loc[idx, feature_column] = predicted_value

    return df


# 逐个特征列填补空缺值
for feature_column in feature_columns:
    filled_data = fill_missing_values(filled_data, feature_column)

# 保存填充后的数据为CSV文件
output_file = "C:\\Users\\HE CONG BING\\Desktop\\1001_r_all.csv"
filled_data.to_csv(output_file, index=False, float_format="%.1f")  # 不保存索引
