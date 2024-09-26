import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# data = pd.read_csv("..\\data\\Air Quality\\1001.csv")
# PM25 = data['PM25_Concentration'].values
# data = pd.read_csv("..\\data\\Energy\\energydata_complete.csv")
# PM25 = data['Appliances'].values
# data = pd.read_csv("..\\data\\Stocks\\NFLX.csv")
# PM25 = data['High'].values
# PM25_d = np.diff(PM25)
data = pd.read_csv("..\\data\\traffic\\traffic.csv")
PM25 = data['traffic'].values
# PM25_d = np.diff(PM25)
# PM = np.zeros_like(PM25)
# PM[0] = PM25[0]
# PM[1:] = PM25_d
# PM25 = PM
print(len(PM25))


def plot_acf_pacf(ts, max_lags):
    sm.graphics.tsa.plot_acf(ts, lags=max_lags)
    plt.title("自相关函数 (ACF)")

    sm.graphics.tsa.plot_pacf(ts, lags=max_lags)
    plt.title("偏自相关函数 (PACF)")
    plt.show()


def dm_test(data):
    result = adfuller(data)

    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]

    print(f'ADF Statistic: {adf_statistic}')
    print(f'p-value: {p_value}')
    print('Critical Values:')
    for key, value in critical_values.items():
        print(f'   {key}: {value}')

    if p_value <= 0.05:
        print('拒绝原假设，时间序列是平稳的。')
    else:
        print('无法拒绝原假设，时间序列可能是非平稳的。')


def lb_test(data):
    result_lb = acorr_ljungbox(data, lags=[10], return_df=True)
    print(result_lb)
    test_statistic = result_lb['lb_stat']
    p_values = result_lb['lb_pvalue']
    print('p_values:', p_values)

    print(f'Ljung-Box Test Statistic: {test_statistic}')
    print('p-values:')
    for lag, (lb_stat, p_value) in enumerate(zip(test_statistic, p_values)):
        print(f'   Lag {lag + 1}: lb_stat = {lb_stat}, p-value = {p_value}')

    alpha = 0.05
    rejected = p_values < alpha
    if any(rejected):
        print('拒绝原假设，时间序列存在自相关性。')
    else:
        print('无法拒绝原假设，时间序列不存在自相关性。')


def jb_test(data):
    jb_statistic, p_value, _, _ = sm.stats.stattools.jarque_bera(data)

    print(f'Jarque-Bera Test Statistic: {jb_statistic}')
    print(f'p-value: {p_value}')

    alpha = 0.05
    if p_value < alpha:
        print('拒绝原假设，数据不服从正态分布。')
    else:
        print('无法拒绝原假设，数据可能服从正态分布。')


if __name__ == '__main__':
    plot_acf_pacf(PM25, 500)
    dm_test(PM25)
    lb_test(PM25)
    jb_test(PM25)
