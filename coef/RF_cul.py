import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
# from data_loader_rf_all import train_data, train_targets
from data_loader_rf_all import train_data, train_targets, test_data, test_targets

plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

rf = RandomForestRegressor(n_estimators=300)

rf.fit(train_data, train_targets)

y_pred = rf.predict(test_data)
mse = mean_squared_error(test_targets, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_targets, y_pred)

print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(test_targets, label='真实PM2.5含量', color='red', linewidth=2)
plt.plot(y_pred, label='预测PM2.5含量', linestyle='--', color='blue', linewidth=2)
plt.xlabel('时间')
plt.ylabel('PM2.5含量')
plt.legend()
plt.show()

importances = rf.feature_importances_

feature_names = ["PM25_Concentration", "PM10_Concentration", "NO2_Concentration", "CO_Concentration",
                 "O3_Concentration", "SO2_Concentration", "weather", "temperature", "pressure", "humidity",
                 "wind_speed", "wind_direction", "time_num", "date_num", "PM25_average", "PM25_median", "PM25_max",
                 "PM25_min", "PM25_std", "PM25_coef"]

for feature_name, importance in zip(feature_names, importances):
    print(f"{feature_name}: {importance:.4f}")

sorted_indices = np.argsort(importances)
sorted_importances = importances[sorted_indices]
sorted_feature_names = [feature_names[i] for i in sorted_indices]

plt.figure(figsize=(10, 6))
colors = ['#{:02x}{:02x}{:02x}'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in
          range(len(sorted_importances))]
plt.barh(range(len(sorted_importances)), sorted_importances, tick_label=sorted_feature_names, color=colors)
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance (Sorted)')
plt.gca().invert_yaxis()
plt.show()
