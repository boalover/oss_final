from transformers import pipeline

def sentiment_analysis(ticker):
    """Perform sentiment analysis on recent news headlines."""
    sentiment_pipeline = pipeline("sentiment-analysis")
    example_headlines = [
        f"{ticker} sees significant growth in recent quarter",
        f"{ticker} faces challenges amid economic downturn"
    ]
    sentiments = [sentiment_pipeline(headline)[0]['label'] for headline in example_headlines]
    sentiment_score = sum(1 if sentiment == 'POSITIVE' else -1 for sentiment in sentiments)
    return sentiment_score
