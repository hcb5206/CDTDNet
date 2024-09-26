import matplotlib.pyplot as plt
from data_loader_coef_f import train_data, train_targets, train_data_out
from data_loader_coef_t import train_inputs_s, train_targets_s, train_data_out_s
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

# spearman相关系数获取特征的非线性相关性
data = pd.DataFrame(train_data_out,
                    columns=["PM25_Concentration", "PM10_Concentration", "NO2_Concentration", "CO_Concentration",
                             "O3_Concentration", "SO2_Concentration", "weather", "temperature", "pressure", "humidity",
                             "wind_speed", "wind_direction", "time_num", "date_num", "Value_Lag_1", "Value_Lag_2",
                             "Value_Diff", "Target"])

correlations = {}
for feature in data.columns[:-1]:
    corr, _ = spearmanr(data[feature], data["Target"])
    correlations[feature] = corr

sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
sorted_features = [x[0] for x in sorted_correlations]
sorted_importances = [x[1] for x in sorted_correlations]

plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_importances, y=sorted_features, orient="h")
plt.xlabel('Spearman Rank Correlation Coefficient (Absolute Value)')
plt.ylabel('Features')
plt.title('Feature Importance Based on Spearman Rank Correlation')
plt.show()

# Pearson相关系数获取特征重要性
data = pd.DataFrame(train_data_out_s,
                    columns=["PM25_average", "PM25_median", "PM25_max", "PM25_min", "PM25_std", "PM25_coef", "Target"])

correlations = {}
for feature in data.columns[:-1]:
    corr = data[feature].corr(data["Target"])
    correlations[feature] = corr

sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
sorted_features = [x[0] for x in sorted_correlations]
sorted_importances = [x[1] for x in sorted_correlations]

plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_importances, y=sorted_features, orient="h")
plt.xlabel('Pearson Correlation Coefficient (Absolute Value)')
plt.ylabel('Features')
plt.title('Feature Importance Based on Pearson Correlation')
plt.show()
