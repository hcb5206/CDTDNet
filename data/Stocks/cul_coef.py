import matplotlib.pyplot as plt
from data.Stocks.data_loader_coef_f_stock import train_data_out
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

# spearman相关系数获取特征的非线性相关性
# data = pd.DataFrame(train_data_out,
#                     columns=["AAL", "AAPL", "ADBE", "ADI", "ADP", "ADSK", "AKAM", "ALXN", "AMAT", "AMGN", "AMZN",
#                              "ATVI", "AVGO", "BBBY", "BIDU", "BIIB", "CA", "CELG", "CERN", "CMCSA", "COST", "CSCO",
#                              "CSX", "CTRP", "CTSH", "DISCA", "DISH", "DLTR", "EA", "EBAY", "ESRX", "EXPE", "FAST", "FB",
#                              "FOX", "FOXA", "GILD", "GOOGL", "INTC", "JD", "KHC", "LBTYA", "LBTYK", "LRCX", "MAR",
#                              "MAT", "MCHP", "MDLZ", "MSFT", "MU", "MXIM", "MYL", "NCLH", "NFLX", "NTAP", "NVDA", "NXPI",
#                              "PAYX", "PCAR", "PYPL", "QCOM", "QVCA", "ROST", "SBUX", "SIRI", "STX", "SWKS", "SYMC",
#                              "TMUS", "TRIP", "TSCO", "TSLA", "TXN", "VIAB", "VOD", "VRTX", "WBA", "WDC", "WFM", "XLNX",
#                              "YHOO", "NDX", "Value_Lag_1",  "Value_Lag_2", "Value_Diff", "Target"])

data = pd.DataFrame(train_data_out,
                    columns=["AAL", "AAPL", "ADBE", "ADI", "ADP", "ADSK", "AKAM", "ALXN", "AMAT", "AMGN", "AMZN",
                             "ATVI", "AVGO", "BBBY", "BIDU", "BIIB", "CA", "CELG", "CERN", "CMCSA", "COST", "CSCO",
                             "Value_Lag_1", "Value_Lag_2", "Value_Diff", "Target"])

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
