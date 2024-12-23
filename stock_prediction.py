import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tkinter as tk
from tkinter import messagebox, Toplevel

# 리더보드 기능
leaderboard = {}

def update_leaderboard(user_name, predicted_growth):
    """리더보드 업데이트"""
    leaderboard[user_name] = predicted_growth

def display_leaderboard():
    """리더보드 출력"""
    sorted_leaderboard = sorted(leaderboard.items(), key=lambda x: x[1], reverse=True)
    
    leaderboard_window = Toplevel()  # 새로운 창 생성
    leaderboard_window.title("Leaderboard")
    
    tk.Label(leaderboard_window, text="Leaderboard", font=("Arial", 16)).grid(row=0, column=0, columnspan=2, padx=10, pady=10)
    
    for rank, (user, growth) in enumerate(sorted_leaderboard, 1):
        tk.Label(leaderboard_window, text=f"{rank}. {user}: {growth * 100:.2f}%", font=("Arial", 12)).grid(row=rank, column=0, padx=10, pady=5)

    # "Back to Input" 버튼 추가
    def go_back_to_input():
        leaderboard_window.destroy()  # 리더보드 창 닫기
        run_input_window()  # 입력 창 다시 실행

    back_button = tk.Button(leaderboard_window, text="Back to Input", command=go_back_to_input)
    back_button.grid(row=len(sorted_leaderboard) + 1, column=0, columnspan=2, pady=10)
    
    leaderboard_window.mainloop()

# 데이터 전처리 함수
def preprocess_data(data, scaler_type='minmax'):
    """데이터 스케일링"""
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    return scaled_data, scaler

def create_dataset(dataset, time_step):
    """LSTM 입력 데이터 생성"""
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:i + time_step])
        y.append(dataset[i + time_step])
    return np.array(X), np.array(y)

# LSTM 모델 빌드
def build_lstm_model(input_shape):
    """LSTM 모델 생성 (개선된 버전)"""
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(input_shape, 1)))
    model.add(Dropout(0.3))  # Dropout 추가
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.3))  # Dropout 추가
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 감정 분석 (간단한 임시 함수)
def sentiment_analysis(ticker):
    """주식 감정 점수 (임의로 구현)"""
    sentiment_score = np.random.choice([-1, 0, 1])  # 부정적(-1), 중립(0), 긍정적(+1)
    return sentiment_score

# 주식 데이터 가져오기
def fetch_stock_data(ticker, start_date, end_date):
    """Yahoo Finance에서 주식 데이터 가져오기"""
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# 주식 예측 메인 함수
def predict_stock_price(ticker, start_date, end_date, user_name, future_days=30):
    """주식 예측 실행"""
    # 데이터 가져오기
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    close_prices = stock_data['Close']
    dates = stock_data.index  # 날짜 가져오기

    # 데이터 전처리
    scaled_data, scaler = preprocess_data(close_prices, scaler_type='standard')
    time_step = 60
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # 학습/테스트 데이터 분리
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # LSTM 모델 학습
    model = build_lstm_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)  # epochs 30으로 증가

    # 예측 수행 (X_test에서 예측)
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 감정 점수 조정
    sentiment_score = sentiment_analysis(ticker)
    sentiment_adjustment = sentiment_score * 0.001  # 감정 분석 영향력을 줄임
    predictions_adjusted = predictions * (1 + sentiment_adjustment)

    # 예측 날짜 범위 설정
    forecast_dates = dates[train_size + time_step:]

    # **연속적인 예측 (미래 예측)**
    last_data = scaled_data[-time_step:]  # 마지막 데이터 포인트
    future_predictions = []

    # 미래 예측 (future_days)
    for i in range(future_days):
        # 예측하고 그 예측값을 새로운 입력 데이터로 사용
        future_input = last_data.reshape(1, time_step, 1)
        future_price = model.predict(future_input)
        future_price = scaler.inverse_transform(future_price)

        # 예측값을 저장하고 다음 입력을 준비
        future_predictions.append(future_price[0, 0])
        last_data = np.append(last_data[1:], future_price[0, 0])

    # 미래 예측 날짜 생성
    future_dates = [dates[-1] + np.timedelta64(i+1, 'D') for i in range(future_days)]

    # 결과 시각화
    plt.figure(figsize=(14, 5))
    plt.plot(dates[train_size + time_step:], y_test, label='Actual Price', color='blue')
    plt.plot(forecast_dates, predictions, label='Predicted Price', color='green')
    plt.plot(forecast_dates, predictions_adjusted, label='Sentiment Adjusted Price', color='red')
    plt.plot(future_dates, future_predictions, label='Future Predicted Price', color='orange', linestyle='dashed')

    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

    # 리더보드 업데이트
    predicted_growth = (predictions_adjusted[-1] - predictions_adjusted[0]) / predictions_adjusted[0]
    update_leaderboard(user_name, predicted_growth)

    # 리더보드 출력
    display_leaderboard()

# GUI 구현
def run_input_window():
    """주식 입력 창"""
    def on_submit():
        user_name = entry_user_name.get()
        ticker = entry_ticker.get()
        start_date = entry_start_date.get()
        end_date = entry_end_date.get()

        if not user_name or not ticker or not start_date or not end_date:
            messagebox.showerror("Input Error", "Please fill in all fields.")
        else:
            predict_stock_price(ticker, start_date, end_date, user_name, future_days=30)

    root = tk.Tk()
    root.title("Stock Prediction GUI")

    # 사용자 이름 입력
    tk.Label(root, text="Enter your username:").grid(row=0, column=0, padx=10, pady=5)
    entry_user_name = tk.Entry(root)
    entry_user_name.grid(row=0, column=1, padx=10, pady=5)

    # 주식 티커 입력
    tk.Label(root, text="Enter stock ticker (e.g., AAPL):").grid(row=1, column=0, padx=10, pady=5)
    entry_ticker = tk.Entry(root)
    entry_ticker.grid(row=1, column=1, padx=10, pady=5)

    # 시작일 입력
    tk.Label(root, text="Enter start date (YYYY-MM-DD):").grid(row=2, column=0, padx=10, pady=5)
    entry_start_date = tk.Entry(root)
    entry_start_date.grid(row=2, column=1, padx=10, pady=5)

    # 종료일 입력
    tk.Label(root, text="Enter end date (YYYY-MM-DD):").grid(row=3, column=0, padx=10, pady=5)
    entry_end_date = tk.Entry(root)
    entry_end_date.grid(row=3, column=1, padx=10, pady=5)

    # 제출 버튼
    submit_button = tk.Button(root, text="Submit", command=on_submit)
    submit_button.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

    # 리더보드 버튼
    leaderboard_button = tk.Button(root, text="View Leaderboard", command=display_leaderboard)
    leaderboard_button.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

    root.mainloop()

# GUI 실행
if __name__ == "__main__":
    run_input_window()
