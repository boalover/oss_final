import yfinance as yf

def calculate_growth_rate(ticker, start_date, end_date):
    """Calculate the growth rate of the stock based on historical data."""
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    start_price = stock_data['Close'][0]
    end_price = stock_data['Close'][-1]
    return (end_price - start_price) / start_price * 100
