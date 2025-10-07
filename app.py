from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

@app.route('/')
def home():
    return render_template('index.html')  # Your HTML file in templates/

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if not text.strip():
        return jsonify({'sentiment': 'No input provided'})

    result = sentiment_pipeline(text)[0]
    return jsonify({'sentiment': f"{result['label']} ({round(result['score'] * 100, 2)}%)"})

if __name__ == '__main__':
    app.run(debug=True)