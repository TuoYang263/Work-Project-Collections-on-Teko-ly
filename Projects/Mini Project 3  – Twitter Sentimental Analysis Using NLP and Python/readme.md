# Twitter Sentiment Analysis Using NLP and Python

## Overview
This project leverages Natural Language Processing (NLP) and machine learning techniques to analyze sentiments expressed in tweets. By processing text data and utilizing Python's extensive libraries, the project aims to classify tweets into three sentiment categories: Neutral, Negative, and Positive.

Twitter, as one of the largest platforms for textual expression, provides valuable data for sentiment analysis. By analyzing this data, businesses and organizations can derive meaningful insights, such as understanding customer opinions or product feedback.

---

## Objective
- To classify tweets into sentiment categories using NLP techniques and machine learning.
- Utilize Python libraries for data processing, visualization, and modeling, including:
  - **Pandas**: Data manipulation.
  - **Seaborn/Matplotlib**: Data visualization.
  - **NLTK**: Text cleaning and NLP tasks.
  - **TensorFlow/Keras**: Model building using LSTM.
  - **Sklearn**: Performance metrics and classification report.

---

## Dataset Description
The dataset contains two columns:
- **Tweets**: Textual data written by individuals.
- **Category**: Sentiment type represented as:
  - `0` for Neutral.
  - `-1` for Negative.
  - `1` for Positive.

---

## Tasks Performed
1. **Data Preparation**:
   - Loaded the dataset from an Excel file.
   - Transformed the dependent variable (`Category`) into categorical labels (`Neutral`, `Negative`, `Positive`).
   - Conducted missing value analysis and removed null values.

2. **Text Cleaning**:
   - Removed non-alphanumeric characters, punctuation, and stopwords.
   - Converted all text to lowercase.

3. **Feature Engineering**:
   - Created a new column to calculate the length of each tweet (word count).

4. **Data Splitting**:
   - Split data into independent variables (`X`) and dependent variables (`y`).

5. **Text Data Operations**:
   - Applied one-hot encoding for tokenized sentences.
   - Added padding to ensure uniform input length.

6. **Model Building**:
   - Designed and compiled an LSTM model with:
     - **Embedding layer**: Converts tokens to dense vector representations.
     - **LSTM layer**: Captures sequential relationships in text data.
     - **Dropout layer**: Reduces overfitting.
     - **Output layer**: Activation function suited for categorical classification.
   - Performed dummy variable creation for the dependent variable.

7. **Model Training**:
   - Split the dataset into training and test sets.
   - Trained the LSTM model using the training data.

8. **Evaluation**:
   - Normalized predictions to match the original categories.
   - Evaluated the model using classification reports and accuracy metrics.

---

## Results
- Achieved a high accuracy rate in classifying tweets into sentiment categories.
- Generated classification reports to evaluate precision, recall, and F1 scores for each sentiment class.

---

## File Structure
```plaintext
Twitter Sentimental Analysis Using NLP/
├── scripts/
    └── Mini Project 3 – Twitter Sentimental Analysis  Using NLP and Python (Python scripts).py
├── Mini Project 3 – Twitter Sentimental Analysis  Using NLP and Python.ipynb
├── model.png
├── best_model/
    └── best_model.keras
├── dataset/
    └── Twitter_Data.csv
├── graphviz_installation_package/
    └── download_needed.txt
├── tokenized_data/
    └── tokenizer_data.pickle