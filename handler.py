from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
import torch

# Load the label encoder
label_encoder = joblib.load('label_encoder.pkl')

# Load the model and tokenizer from Hugging Face
model_name = "SCANSKY/distilbertTourism-multilingual-sentiment"
sentiment_analyzer = pipeline(
    'sentiment-analysis',
    model=model_name,
    tokenizer=model_name,
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available
)

def get_average_sentiment(positive_count, negative_count, neutral_count):
    total = positive_count + negative_count + neutral_count
    if total == 0:
        return "neutral"
    
    positive_pct = (positive_count / total) * 100
    negative_pct = (negative_count / total) * 100
    neutral_pct = (neutral_count / total) * 100
    
    max_sentiment = max(positive_pct, negative_pct, neutral_pct)
    
    if max_sentiment == positive_pct:
        return "positive"
    elif max_sentiment == negative_pct:
        return "negative"
    else:
        return "neutral"

class Handler:
    def __init__(self):
        # Model and tokenizer are loaded globally, so no need to reinitialize here
        pass

    def preprocess(self, data):
        # Extract the input text from the request
        text = data.get("text", "")
        return text

    def inference(self, text):
        if not text.strip():
            return {"error": "Please enter some text for sentiment analysis."}
        
        # Split text into lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            return {"error": "Please enter valid text for sentiment analysis."}
        
        # Analyze each line
        total_confidence = 0
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        line_results = []  # Store results for each line
        
        for line in lines:
            result = sentiment_analyzer(line)
            predicted_label_encoded = int(result[0]['label'].split('_')[-1])
            predicted_label = label_encoder.inverse_transform([predicted_label_encoded])[0]
            confidence = result[0]['score'] * 100
            
            # Store line and its sentiment result
            line_results.append({
                'text': line,
                'sentiment': predicted_label,
                'confidence': confidence
            })
            
            if predicted_label == 'positive':
                positive_count += 1
            elif predicted_label == 'negative':
                negative_count += 1
            else:
                neutral_count += 1
            
            total_confidence += confidence
        
        # Calculate averages
        avg_confidence = total_confidence / len(lines)
        positive_pct = (positive_count / len(lines)) * 100
        negative_pct = (negative_count / len(lines)) * 100
        neutral_pct = (neutral_count / len(lines)) * 100
        
        # Get average sentiment
        avg_sentiment = get_average_sentiment(positive_count, negative_count, neutral_count)
        
        # Prepare the output
        output = {
            "total_lines_analyzed": len(lines),
            "average_confidence": avg_confidence,
            "average_sentiment": avg_sentiment,
            "sentiment_distribution": {
                "positive": positive_pct,
                "negative": negative_pct,
                "neutral": neutral_pct
            },
            "line_results": line_results
        }
        
        return output

    def postprocess(self, output):
        # Format the output for the response
        if "error" in output:
            return {"error": output["error"]}
        
        return {
            "results": output
        }

    def handle(self, data):
        # Main method to handle the request
        text = self.preprocess(data)
        output = self.inference(text)
        return self.postprocess(output)