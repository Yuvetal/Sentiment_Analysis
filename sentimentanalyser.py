from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Example usage
text = "I am not feeling well today."
result = sentiment_pipeline(text)[0]

print(f"Sentiment: {result['label']} ({round(result['score'] * 100, 2)}%)")
