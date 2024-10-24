

# Sentiment Analysis for Reviews taken from Kaggle Test set

## Overview

This project performs sentiment analysis on Amazon product reviews using a Large Language Model (LLM). It processes reviews in parallel, classifies them as either "Good" or "Bad", and provides accuracy metrics and visualizations.

## Features

- Parallel processing of reviews for faster analysis
- Sentiment classification using a custom LLM API
- Accuracy calculation and confusion matrix visualization
- Progress bar for real-time processing updates
- Token usage tracking
- CSV output of results

## Requirements

- Python 3.7+
- Required Python packages:
  - requests
  - matplotlib
  - seaborn
  - scikit-learn
  - gradio
  - tqdm

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/satyaNekkantiCompVison/reviews_analyze.git
   cd reviews_analyze
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your LLM API endpoint in the `get_sentiment` function within `review_gt_compare.py`.

## Usage

1. Prepare your input file with Amazon reviews. Each line should be in the format:
   ```
   __label__1 Review text here
   ```
   or
   ```
   __label__2 Review text here
   ```
   Where `__label__1` represents a negative review and `__label__2` represents a positive review.

   or Download the data from this link https://www.kaggle.com/datasets/bittlingmayer/amazonreviews?resource=download
   

3. Run the script:
   ```
   python review_gt_compare.py
   ```

4. The script will process the reviews and output:
   - A CSV file with the results (`sentiment_analysis_results.csv`)
   - A confusion matrix image (`confusion_matrix.png`)
   - Console output with accuracy metrics and processing statistics

## Customization

- Adjust the `max_reviews` parameter in `process_reviews` to change the number of reviews processed.
- Modify the `max_workers` parameter in `ThreadPoolExecutor` to adjust the level of parallelism.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
