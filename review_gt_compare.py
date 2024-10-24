import csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import requests
import json
import gradio as gr
import time
from tqdm import tqdm
import concurrent.futures

def clean_sentiment(sentiment):
    # print(sentiment)
    if 'good' in sentiment.lower():
        return 'Good'
    elif 'bad' in sentiment.lower():
        return 'Bad'
    else:
        return 'Unknown'

def get_sentiment_with_comments(review):
    url = "http://192.168.0.107:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "messages": [
            {"role": "system", "content": "You are a sentiment analysis expert. Classify the following review as 'Good' or 'Bad'. If you have any additional comments, provide them separately and consider works fine also a good review"},
            {"role": "user", "content": review}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    result = response.json()
    if 'choices' in result and len(result['choices']) > 0:
        content = result['choices'][0]['message']['content'].strip()
        parts = content.split('\n', 1)
        sentiment = clean_sentiment(parts[0])
        comments = parts[1] if len(parts) > 1 else ""
        return sentiment, comments.strip()
    else:
        return "Error", "Unable to get sentiment"

def get_sentiment(review):
    url = "http://192.168.0.107:1234/v1/chat/completions"  # Adjust the URL if necessary
    headers = {"Content-Type": "application/json"}
    data = {
        "messages": [
            {"role": "system", "content": "You are a sentiment analysis expert. Classify the following review as 'Good' or 'Bad'."},
            {"role": "user", "content": review}
        ],
        "temperature": 0.7,
        "max_tokens": 50
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    result = response.json()
    if 'choices' in result and len(result['choices']) > 0:
        sentiment = clean_sentiment(result['choices'][0]['message']['content'].strip())
        # Extract token usage from the response
        prompt_tokens = result.get('usage', {}).get('prompt_tokens', 0)
        completion_tokens = result.get('usage', {}).get('completion_tokens', 0)
        total_tokens = prompt_tokens + completion_tokens
        return sentiment, total_tokens
    else:
        return "Error: Unable to get sentiment", 0

def process_review(review_data):
    i, (label, review) = review_data
    sentiment, tokens = get_sentiment(review)
    return f"{label}_{i}", 'Bad' if label == '__label__1' else 'Good', review, sentiment, tokens

def process_reviews(input_file, output_file, max_reviews=100):
    reviews = []
    ground_truth = []
    predictions = []
    total_tokens = 0
    start_time = time.time()

    # Read input file and process reviews
    with open(input_file, 'r', encoding='utf-8') as f:
        reviews = [(line.strip().split(' ', 1)[0], line.strip().split(' ', 1)[1]) for line in f][:max_reviews]
        ground_truth = ['Bad' if label == '__label__1' else 'Good' for label, _ in reviews]

    # Process reviews in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(process_review, enumerate(reviews)), total=len(reviews), desc="Processing reviews"))

    # Save results to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'Original Label', 'Review', 'Predicted Sentiment', 'Tokens Used'])
        for result in results:
            writer.writerow(result)
            predictions.append(result[3])  # Sentiment
            total_tokens += result[4]  # Tokens

    # Calculate total time
    total_time = time.time() - start_time
    total_minutes = total_time / 60
    total_hours = total_minutes / 60

    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Total time taken: {total_minutes:.2f} minutes")
    print(f"Total time taken: {total_hours:.2f} hours")
    print(f"Total tokens used: {total_tokens}")

    # Calculate accuracy
    # Convert 'Good' and 'Bad' to numerical values for comparison
    ground_truth_numeric = [1 if label == 'Good' else 0 for label in ground_truth]
    predictions_numeric = [1 if pred == 'Good' else 0 for pred in predictions]
    accuracy = accuracy_score(ground_truth_numeric, predictions_numeric)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Print some debug information
    print(f"Ground Truth (first 10): {ground_truth[:10]}")
    print(f"Predictions (first 10): {predictions[:10]}")
    print(f"Total samples: {len(ground_truth)}")
    print(f"Correct predictions: {sum([1 for gt, pred in zip(ground_truth, predictions) if gt == pred])}")

    # Create confusion matrix
    cm = confusion_matrix(ground_truth, predictions, labels=['Good', 'Bad'])
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
    plt.title('Confusion Matrix: Ground Truth vs Predicted')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.savefig('confusion_matrix.png')
    plt.close()

# Usage
input_file = 'Amazon_test_dataset_review.txt'  # Replace with your input file name
output_file = 'sentiment_analysis_results.csv'
process_reviews(input_file, output_file, max_reviews=1000)
