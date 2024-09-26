import numpy as np
import matplotlib.pyplot as plt
from data_loader_rf import train_data

num_samples = train_data.shape[0]
num_features = train_data.shape[1]

sample_entropies = []
for feature in range(num_features):
    feature_values = train_data[:, feature]
    unique_values, value_counts = np.unique(feature_values, return_counts=True)
    probabilities = value_counts / num_samples
    entropy = -np.sum(probabilities * np.log2(probabilities))
    sample_entropies.append(entropy)

plt.figure(figsize=(10, 8))
plt.plot(range(num_features), sample_entropies, marker='o', linestyle='-')
plt.xlabel('Input Features')
plt.ylabel('Sample Entropy')
plt.title('Sample Entropy of Input Features')
plt.xticks(range(num_features), ["PM25_Concentration", "PM10_Concentration", "NO2_Concentration", "CO_Concentration",
                                 "O3_Concentration", "SO2_Concentration", "weather", "temperature", "pressure",
                                 "humidity", "wind_speed", "wind_direction", "time_num", "date_num"], rotation=45)
plt.grid(True)
plt.show()
