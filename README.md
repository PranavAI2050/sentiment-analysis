# ✈️ US Airline Twitter Sentiment Analysis

This project performs sentiment analysis on tweets directed at major US airlines using traditional machine learning and deep learning techniques.

## 📑 Dataset

- **Source**: [Twitter US Airline Sentiment Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- **Classes**: Positive, Neutral, Negative
- **Size**: 14,640 tweets

## 🧹 Preprocessing

- Removed mentions, URLs, hashtags, special characters
- Converted to lowercase
- Removed stopwords
- Cleaned tweets stored as `clean_text`

## 🔍 Exploratory Data Analysis (EDA)

- **Class Distribution**: Imbalanced (mostly negative tweets)
- **Word Count**: Most tweets between 5–15 words
- **Top Words**:
  - Positive: `thank`, `great`, `awesome`, `love`
  - Negative: `cancelled`, `delayed`, `worst`, `hold`
  - Neutral: `ticket`, `flight`, `dm`, `help`

## 🧠 Models Implemented

### 🔹 TF-IDF + Logistic Regression
- `TfidfVectorizer` used to transform text
- Achieved **~76.2% accuracy**
- Precision, Recall, F1-score computed for each class

### 🔹 BiLSTM (Bidirectional LSTM)
- Keras model using embedding layer + BiLSTM
- Better class balance performance, especially for positive/neutral
- Achieved **~75% accuracy**

## 📊 Evaluation Metrics

| Model            | Accuracy | F1-Score (Macro Avg) |
|------------------|----------|----------------------|
| TF-IDF + LR      | 76.2%    | 66.6%                |
| BiLSTM           | 75.0%    | 67.0%                |

## 📌 Conclusion

- TF-IDF performed well on negative tweets but struggled with neutral.
- BiLSTM provided more balanced performance across all sentiments.
- Room for improvement by using:
  - Pretrained embeddings (GloVe)
  - Transformer models (BERT)
  - Class imbalance handling techniques

