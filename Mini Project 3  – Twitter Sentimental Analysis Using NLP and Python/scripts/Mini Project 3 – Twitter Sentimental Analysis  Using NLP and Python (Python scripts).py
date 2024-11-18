#!/usr/bin/env python
# coding: utf-8

# In[40]:


# import necessary libraries
import pandas as pd
import seaborn as sns
import nltk
import nltk.corpus
import re
import string
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split


# In[41]:


get_ipython().system('pip install wordcloud')
get_ipython().system('pip install pydot==2.0.0')
get_ipython().system('pip install graphviz pydot')
get_ipython().system('pip install pydot-ng')


# In[42]:


import os
os.environ["PATH"] += os.pathsep + 'C:/Graphviz/bin'


# In[43]:


# check if Python Graphviz and Pydot Libraries are successfully installed
import pydot
pydot.Dot.create(pydot.Dot())


# #### Read the Data from the Given excel file.

# In[44]:


twitter_data = pd.read_csv("dataset/Twitter_Data.csv")


# Check the first five rows of the data

# In[45]:


twitter_data.head()


# Check the size of the dataset

# In[46]:


twitter_data.shape


# check column attribute info

# In[47]:


twitter_data.info()


# #### Do Missing value analysis and drop all null/missing values

# Check if null values exist in the columns

# In[48]:


twitter_data.isna().sum()


# In[49]:


twitter_data = twitter_data.dropna(subset=['clean_text', 'category'], axis=0)


# Check if null values are still there after missing values processing

# In[50]:


twitter_data.isna().sum()


# #### Change our dependent variable to categorical. ( 0 to “Neutral,” -1 to “Negative”, 1 to “Positive”)

# In[51]:


twitter_data['category'] = twitter_data['category'].replace({0: "Neutral", -1: "Negative", 1: "Positive"})


# In[52]:


twitter_data.head()


# Display the percentage of target variable values

# In[53]:


twitter_categ_counts = twitter_data['category'].value_counts()
twitter_categ_perc = (twitter_categ_counts/twitter_categ_counts.sum()) * 100
twitter_categ_perc_df = twitter_categ_perc.reset_index()
twitter_categ_perc_df.columns = ['Category', 'Percentage']
twitter_categ_perc_df


# #### Create a new column and find the length of each sentence (how many words they contain)

# In[54]:


twitter_data['word_count'] = twitter_data['clean_text'].apply(lambda x: len(x.split()))


# In[55]:


# check the column 'word_count'
twitter_data.head()


# #### Exploratory Data Analysis

# Data visualization, check the distribution of sentiments (target variables)

# In[56]:


# Ref: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html
twitter_data.drop(['word_count'], axis=1).groupby('category').count().plot(kind='bar',
                                              rot='horizontal',
                                              title='The Distribution of Sentiments')


# Plot the distribution of text length for positive sentiment tweets

# In[57]:


# Plotting the distribution of tweet text lengths
text_len = pd.Series([len(text.split()) for text in twitter_data['clean_text']])

# The distribution of tweet lengths 
text_len.plot(kind='box', xlabel='Tweet Text', ylabel='Tweet Text Length', title='The Distribution of Tweet Text Lengths')
plt.show()


# Plot the distribution of text length for positive sentiment tweets

# In[58]:


fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(121)
sns.histplot(twitter_data[twitter_data['category'] == 'Positive']['word_count'], ax=ax1, color='blue')

describe = pd.DataFrame(twitter_data[twitter_data['category'] == 'Positive']['word_count'].describe()).round(2)
ax2 = fig.add_subplot(122)
# close all axes of in the 2nd figure
ax2.axis('off')
font_size = 12
# bbox: coordinates of the left lower and left upper points
bbox = [0, 0, 1, 1]
table = ax2.table(cellText=describe.values, rowLabels=describe.index, colLabels=describe.columns, bbox=bbox)
table.set_fontsize(font_size)
fig.suptitle('The Distribution of Text Length for Positive Sentiment Tweets.', fontsize=14)
plt.show()


# Plot the distribution of text length for negative sentiment tweets

# In[59]:


fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(121)
sns.histplot(twitter_data[twitter_data['category'] == 'Negative']['word_count'], ax=ax1, color='red')

describe = pd.DataFrame(twitter_data[twitter_data['category'] == 'Negative']['word_count'].describe()).round(2)
ax2 = fig.add_subplot(122)
# close all axes of in the 2nd figure
ax2.axis('off')
font_size = 12
# bbox: coordinates of the left lower and left upper points
bbox = [0, 0, 1, 1]
table = ax2.table(cellText=describe.values, rowLabels=describe.index, colLabels=describe.columns, bbox=bbox)
table.set_fontsize(font_size)
fig.suptitle('The Distribution of Text Length for Negative Sentiment Tweets.', fontsize=14)
plt.show()


# Plot the pie chart of the percentage of different sentiments of all the tweets

# In[60]:


import plotly.express as px

senti_percent_fig = px.pie(twitter_data, names="category", title="Pie chart of different sentiments of tweets")
senti_percent_fig.update_layout(margin=dict(t=35, b=10, l=0, r=0))
senti_percent_fig.show()


# Drop the word_count column

# In[61]:


twitter_data = twitter_data.drop(['word_count'], axis=1)
twitter_data.head()


# Visualize data into wordclouds

# In[62]:


from wordcloud import WordCloud, STOPWORDS

def wordcloud_gen(df, category):
    """
    Generate Word Cloud
    Inputs:
        - df: dataframe of tweet dataset
        - category: Positive/Neutral/Negative
    """
    # Combine all the words into a sentence
    combined_texts = ' '.join([text for text in df[df['category']==category]['clean_text']])

    # Initialize wordcloud
    word_cloud = WordCloud(max_words=60,
                           background_color='white',
                           stopwords=STOPWORDS)

    # Generate and plot wordcloud
    plt.figure(figsize=(10, 10))
    plt.imshow(word_cloud.generate(combined_texts))
    plt.title(f'{category} Sentiment Words', fontsize=18)
    plt.axis('off')
    plt.show()

# Positive tweet words
wordcloud_gen(twitter_data, 'Positive')

# Netural tweet words
wordcloud_gen(twitter_data, 'Neutral')

# Negative tweet words
wordcloud_gen(twitter_data, 'Negative')


# #### Do text cleaning. (remove every symbol except alphanumeric, transform all words to lower case, and remove punctuation and stopwords ) 

# In[63]:


"""
A stop word is a commonly used word (such as “the”, “a”, “an”, or “in”) 
that a search engine has been programmed to ignore, 
both when indexing entries for searching and when retrieving them as the result of a search query. 
"""
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from joblib import Parallel, delayed

# setup and initialize the stopwords
stop_words = set(stopwords.words('english'))

def text_cleaning(clean_text):
    """
    Transfer all words to lower case
    """
    text_processed = clean_text.lower()
    
    """
    Remove punctuations
    """
    text_processed = re.sub(r'[^a-zA-Z0-9\s]', '', text_processed)

    """
    Remove stopwords
    """
    word_tokens = word_tokenize(text_processed)
    filtered_sentence = [w for w in word_tokens if w not in stop_words]

    """
    Return a completely sentense
    """
    return " ".join(filtered_sentence)

# Use joblib to parallelly process big amounts of data
demo_original_tweet = twitter_data['clean_text'][0]
twitter_data['clean_text'] = Parallel(n_jobs=-1)(delayed(text_cleaning)(text) for text in twitter_data['clean_text'])
demo_processed_tweet = twitter_data['clean_text'][0]

print("Original text ->", demo_original_tweet)
print("Processed text ->", demo_processed_tweet)


# In[64]:


# check the text data after processing
twitter_data.head()


# #### Split data into independent(X) and dependent(y) dataframe, Do operations on text data

# Tokenizing & Padding

# In[65]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_words = 5000
max_len = 50

def tokenize_pad_sequences(text):
    """
    This function tokenize the input text into sequences of integers and then pad each sequence to the same length
    """
    # Text tokenization
    tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
    tokenizer.fit_on_texts(text)
    # Transform the text to a sequence of integers
    X = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    X = pad_sequences(X, padding='post', maxlen=max_len)
    # return sequences
    return X, tokenizer

print('Text Before Tokeinzation & Padding \n', twitter_data['clean_text'][0])
X, tokenizer = tokenize_pad_sequences(twitter_data['clean_text'])
print('After Tokenization & Padding \n', X[0])


# Save tokenized data

# In[66]:


import pickle

# save
with open('tokenized_data/tokenizer_data.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load
with open('tokenized_data/tokenizer_data.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# Train & Test Split

# In[67]:


# train and test split
y = pd.get_dummies(twitter_data['category'])
# The ratio of training dataset against testing dataset: 80: 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# The ratio of training dataset against validation dataset: 75: 25 ratio
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
print("The Size of Train Set ->", X_train.shape, y_train.shape)
print("The Size of Validation Set ->", X_val.shape, y_val.shape)
print("The Size of Test Set ->", X_test.shape, y_test.shape)


# #### Train new model

# In[68]:


# build bidirectional LSTM Using NN
from keras.models import Sequential
from keras.layers import Input, Embedding, Conv1D, MaxPool1D, Bidirectional, LSTM, Dense, Dropout
from keras.metrics import Precision, Recall
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras import datasets
from keras.callbacks import LearningRateScheduler
from keras.callbacks import History
from keras import losses
from keras.regularizers import l2

vocab_size = 5000
embedding_size = 32
epochs = 20
learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=100000,
    decay_rate=learning_rate/epochs
)
momentum = 0.8
input_shape = (None, 50)

sgd = SGD(learning_rate=lr_schedule, momentum=momentum, nesterov=False)
# Build the model
model = Sequential()
model.add(Embedding(vocab_size, embedding_size))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=2))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.4))
# To prevent overfitting, use L2 regularization to add penalities to the model with large penalities, lambda (penalty coefficient=0.01)
model.add(Dense(3, activation='softmax', kernel_regularizer=l2(0.01)))
model.build(input_shape)


# In[69]:


# display the architecture of the network
tf.keras.utils.plot_model(model, show_shapes=True)


# In[70]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

print(model.summary())

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=sgd, 
               metrics=['accuracy', Precision(), Recall()])

# Train model
batch_size = 64
# Save the best model
checkpoint = ModelCheckpoint('./best_model/best_model.keras',
                             monitor='val_accuracy',
                             save_best_only=True,
                             mode='max')
# Setup early stopping, to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    callbacks=[checkpoint, early_stopping])


# Save the best model with the best validation accuracy for further prediction in the next

# In[71]:


from tensorflow.keras.models import load_model

# Load the best model saved
best_model = load_model('./best_model/best_model.keras')


# #### Measure performance metrics and accuracy. Get Model Accuracy & Loss

# In[72]:


# define the function to calculate the f1 score
import keras.backend as K

def cal_f1_score(precision, recall):
    f1_score = 2 * (precision * recall)/(precision + recall + K.epsilon())
    return f1_score


# In[73]:


# Evaluate the model on the testing dataset
loss, accuracy, precision, recall = best_model.evaluate(X_test, y_test, verbose=0)

# Print metrics
print('Model Accuracy: {:.4f}'.format(accuracy))
print('Model Precision: {:.4f}'.format(precision))
print('Model Recall: {:.4f}'.format(recall))
print('Model F1 Score: {:.4f}'.format(cal_f1_score(precision, recall)))


# In[74]:


def plot_training_history(history):
    """
    Function used for plotting history for accuracy and loss
    """
    fig, ax = plt.subplots(1, 2, figure=(12, 8))
    # The 1st plot
    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend(['train', 'validation'], loc='best')
    
    # The 2nd plot
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend(['train', 'validation'], loc='best')

    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()

plot_training_history(history)


# #### Print Classification Report. Plot Confusion Matrix

# In[75]:


from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(model, X_test, y_test):
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    # use model to do the prediction
    y_pred = model.predict(X_test)
    # compute confusion matrix
    cm = confusion_matrix(np.argmax(np.array(y_test),axis=1), np.argmax(y_pred, axis=1))
    # plot confusion matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, fmt='d', 
                xticklabels=sentiment_classes,
                yticklabels=sentiment_classes)
    plt.title('Confusion matrix', fontsize=16)
    plt.xlabel('Actual label', fontsize=12)
    plt.ylabel('Predicted label', fontsize=12)
    
plot_confusion_matrix(model, X_test, y_test)


# In[76]:


from keras.models import load_model

def predict_category(text):
    
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    max_len=50
    
    # Transforms text to a sequence of integers using a tokenizer object
    x_tokenized = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    x_tokenized = pad_sequences(x_tokenized, padding='post', maxlen=max_len)
    # Do the prediction using the loaded model
    y_tokenized = best_model.predict(x_tokenized).argmax(axis=1)
    # Print the predicted sentiment
    print('The predicted sentiment is', sentiment_classes[y_tokenized[0]])


# In[77]:


predict_category(['I hate that I have to go for work every day'])


# In[78]:


predict_category(['My family totally has three memebers'])


# In[79]:


predict_category(['She might be one of the best persons I have ever seen'])

