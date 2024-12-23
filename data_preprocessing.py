import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):
    """Scale and reshape data for LSTM."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    return scaled_data, scaler

def create_dataset(data, time_step=60):
    """Create dataset for LSTM with a specified time step."""
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)
