import matplotlib.pyplot as plt
from data.Energy.data_loader_coef_f import train_data_out
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

# spearman相关系数获取特征的非线性相关性
data = pd.DataFrame(train_data_out,
                    columns=["Appliances", "lights", "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4", "T5",
                             "RH_5", "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9", "T_out", "Press_mm_hg",
                             "RH_out", "Windspeed", "Visibility", "Tdewpoint", "rv1", "rv2", "Value_Lag_1",
                             "Value_Lag_2", "Value_Diff", "Target"])

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
