from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape):
    """Build an LSTM model for prediction."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(input_shape, 1)),
        Dropout(0.2),  # 첫 번째 Dropout
        LSTM(50, return_sequences=False),
        Dropout(0.3),  # 두 번째 Dropout 비율 증가
        Dense(50),     # Dense 층 유닛 수 증가
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
