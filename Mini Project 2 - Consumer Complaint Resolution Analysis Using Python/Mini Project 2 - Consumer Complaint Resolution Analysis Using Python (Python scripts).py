#!/usr/bin/env python
# coding: utf-8

# ## Mini Project 2 - Consumer Complaint Resolution Analysis Using Python

# Import required libraries

# In[3]:


import math
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore")


# In[4]:


import nltk
nltk.download('wordnet')
nltk.download('stopwords')


# In[5]:


from nltk.corpus import wordnet

try:
    wordnet.ensure_loaded()
    print("WordNet is found!")
except Exception as e:
    print("WordNet is Not Found. Please check your NLTK data path")


# Load given datasets

# In[7]:


# On remote private google drive
consumer_complaints_train_df = pd.read_csv('datasets/Consumer_Complaints_train.csv')
consumer_complaints_test_df = pd.read_csv('datasets/Consumer_Complaints_test.csv')


# Print top 5 records of train dataset

# In[9]:


consumer_complaints_train_df.head()


# Print top 5 records of test dataset

# In[11]:


consumer_complaints_test_df.head()


# **Note: Please note that do all given tasks for test and train both datasets.**

# Print shape of train and test datasets

# In[14]:


# print the shape of train datasets
consumer_complaints_train_df.shape


# In[15]:


# print the shape of test datasets
consumer_complaints_test_df.shape


# Print columns of train and test datasets

# In[17]:


# print columns of train datasets
list(consumer_complaints_train_df.columns)


# In[18]:


# print columns of test datasets
list(consumer_complaints_test_df.columns)


# Check data type for both datasets

# In[20]:


# check the data type of training dataset
consumer_complaints_test_df.info()


# In[21]:


# check the data type of testing dataset
consumer_complaints_test_df.info()


# Print missing values in percentage

# In[23]:


# printing missing values in percentage from training dataset
consumer_train_null_df = pd.DataFrame({'Counts': consumer_complaints_train_df.isnull().sum(),
                                      'Percentage': consumer_complaints_train_df.isnull().sum()/len(consumer_complaints_train_df)})
consumer_train_null_df


# In[24]:


# printing missing values in percentage from testing dataset
consumer_test_null_df = pd.DataFrame({'Counts': consumer_complaints_test_df.isnull().sum(),
                                      'Percentage': consumer_complaints_test_df.isnull().sum()/len(consumer_complaints_test_df)})
consumer_test_null_df


# Drop columns where more than 25% of the data are missing.

# In[26]:


# Drop columns where more than 25% of the data are missing in training set
cols_to_drop_in_train = list(consumer_train_null_df.loc[consumer_train_null_df.Percentage > 0.25].index)
consumer_complaints_train_df.drop(columns=cols_to_drop_in_train, axis=1, inplace=True)

# check columns of training set again after dropping columns
pd.DataFrame({'Counts': consumer_complaints_train_df.isnull().sum(),
              'Percentage': consumer_complaints_train_df.isnull().sum()/len(consumer_complaints_train_df)})


# In[27]:


# Drop columns where more than 25% of the data are missing in testing set
cols_to_drop_in_test = list(consumer_test_null_df.loc[consumer_test_null_df.Percentage > 0.25].index)
consumer_complaints_test_df.drop(columns=cols_to_drop_in_test, axis=1, inplace=True)

# check columns of training set again after dropping columns
pd.DataFrame({'Counts': consumer_complaints_test_df.isnull().sum(),
              'Percentage': consumer_complaints_test_df.isnull().sum()/len(consumer_complaints_test_df)})


# Extract Date, Month, and Year from the "Date Received" Column and create new fields for year, month, and day.
# 
# like, df_train['Year_Received'] = df_train['Date received']........(logic)

# In[29]:


# Extract Date, Month, and Year from the "Date Received" Column in traing set, *.dt is not applicable over here cause the
# target column doesn't belong to datetime type
consumer_complaints_train_df['Day_received'] = pd.DatetimeIndex(consumer_complaints_train_df['Date received']).day
consumer_complaints_train_df['Month_received'] = pd.DatetimeIndex(consumer_complaints_train_df['Date received']).month
consumer_complaints_train_df['Year_received'] = pd.DatetimeIndex(consumer_complaints_train_df['Date received']).year
consumer_complaints_train_df[['Year_received', 'Month_received', 'Day_received']].head()


# In[30]:


# Extract Date, Month, and Year from the "Date Received" Column in testing set, *.dt is not applicable over here cause the
# target column doesn't belong to datetime type
consumer_complaints_test_df['Day_received'] = pd.DatetimeIndex(consumer_complaints_test_df['Date received']).day
consumer_complaints_test_df['Month_received'] = pd.DatetimeIndex(consumer_complaints_test_df['Date received']).month
consumer_complaints_test_df['Year_received'] = pd.DatetimeIndex(consumer_complaints_test_df['Date received']).year
consumer_complaints_test_df[['Year_received', 'Month_received', 'Day_received']].head()


# Convert dates from object type to datetime type

# In[32]:


# the convertion is applied for both training and testing sets
consumer_complaints_train_df['Date received'] = pd.to_datetime(consumer_complaints_train_df['Date received'])
consumer_complaints_train_df['Date sent to company'] = pd.to_datetime(consumer_complaints_train_df['Date sent to company'])

consumer_complaints_test_df['Date received'] = pd.to_datetime(consumer_complaints_test_df['Date received'])
consumer_complaints_test_df['Date sent to company'] = pd.to_datetime(consumer_complaints_test_df['Date sent to company'])


# In[33]:


consumer_complaints_train_df.info()


# Calculate the number of days the complaint was with the company
# 
# create new field with help given logic<br>
# Like, Days held = Date sent to company - Date received

# In[35]:


train_dt_sent = consumer_complaints_train_df['Date sent to company']
train_dt_received = consumer_complaints_train_df['Date received']
test_dt_sent = consumer_complaints_test_df['Date sent to company']
test_dt_received = consumer_complaints_test_df['Date received']
consumer_complaints_train_df['Days held'] = (train_dt_sent - train_dt_received).dt.days
consumer_complaints_test_df['Days held'] = (test_dt_sent - test_dt_received).dt.days


# Convert "Days Held" to Int(above column)

# In[37]:


consumer_complaints_train_df['Days held'] = consumer_complaints_train_df['Days held'].astype(int)
consumer_complaints_test_df['Days held'] = consumer_complaints_test_df['Days held'].astype(int)


# Drop "Date Received","Date Sent to Company","ZIP Code", "Complaint ID"

# In[39]:


consumer_complaints_train_df.drop(columns=["Date received","Date sent to company","ZIP code", "Complaint ID"], axis=1, inplace=True)
consumer_complaints_test_df.drop(columns=["Date received","Date sent to company","ZIP code", "Complaint ID"], axis=1, inplace=True)


# Impute null values in "State" by Mode
# (find mode and replace nan value)

# In[41]:


consumer_complaints_train_df = consumer_complaints_train_df.fillna(value = {"State": consumer_complaints_train_df.State.mode()[0]})
consumer_complaints_test_df = consumer_complaints_test_df.fillna(value = {"State": consumer_complaints_test_df.State.mode()[0]})


# Check Missing Values in the dataset

# In[43]:


# checking missing values of each column in the training dataset
consumer_complaints_train_df.isnull().sum()


# In[44]:


# checking missing values of each column in the testing dataset
consumer_complaints_test_df.isnull().sum()


# Categorize Days into Weeks with the help of 'Days Received'

# In[46]:


# make weeks into float so that it can improve the model's accuracy
# For negative days held, take them as zero
def week_converting(item):
    if item < 8:
        return 1
    elif item >= 8 and item < 16:
        return 2
    elif item >=16 and item < 22:
        return 3
    else:
        return 4

consumer_complaints_train_df['Week_Received'] = consumer_complaints_train_df['Day_received'].apply(lambda item: week_converting(item))
consumer_complaints_test_df['Week_Received'] = consumer_complaints_test_df['Day_received'].apply(lambda item: week_converting(item))


# In[47]:


# Info of the column Day_received has already been converted to the column 'Week_Received', directly drop the column 'Day_received'
consumer_complaints_train_df.drop('Day_received', axis=1, inplace=True)
consumer_complaints_test_df.drop('Day_received', axis=1, inplace=True)


# Print head of train and test dataset and observe

# In[49]:


consumer_complaints_train_df.head()


# In[50]:


consumer_complaints_test_df.head()


# Store data of the disputed consumer in the new data frame as "disputed_cons"

# In[52]:


disputed_cons = consumer_complaints_train_df.loc[consumer_complaints_train_df['Consumer disputed?'] == 'Yes']
disputed_cons.head()


# **The following plotting operation is aimed for training data only (cause only it containes the target variable)**

# Plot bar graph for the total no of disputes with the help of seaborn

# In[55]:


# make counts on target varible
sns.set_theme(rc={'figure.figsize':(6, 5)})
ax = sns.countplot(data=consumer_complaints_train_df, x='Consumer disputed?', width=0.4)
ax.bar_label(ax.containers[0])
ax.set_title('Total No of Disputes')
plt.show()


# **From the figure shown above, we can tell among all 358810 records in the dataset, there are 76172 disputes**

# Plot bar graph for the total no of disputes products-wise with help of seaborn

# In[58]:


sns.set_theme(rc={'figure.figsize':(17, 8)})        # set the context with a specific size
ax = sns.countplot(consumer_complaints_train_df, x='Product', hue='Consumer disputed?')
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.tick_params(axis='x', rotation=45)
ax.set_title('Total No of Disputes by Products')
ax.set_ylabel('Count')
plt.show()


# Plot bar graph for the total no of disputes with Top Issues by Highest Disputes , with help of seaborn

# In[60]:


sns.set_theme(rc={'figure.figsize':(17, 8)})        # set the context with a specific size
ax = sns.countplot(disputed_cons,
                   x='Issue',
                   order=disputed_cons['Issue'].value_counts()[0:10].index,
                   width=0.8
                  )
ax.bar_label(ax.containers[0])
ax.tick_params(axis='x', rotation=70)
ax.set_title('Issues that Having TOP 10 Higest Disputes')
ax.set_ylabel('No of Consumer Disputes')
plt.show()


# Plot bar graph for the total no of disputes by State with Maximum Disputes

# In[62]:


sns.set_theme(rc={'figure.figsize':(17, 8)})        # set the context with a specific size
ax = sns.countplot(disputed_cons,
                   x='State',
                   order=disputed_cons['State'].value_counts()[0:10].sort_values(ascending=False).index,
                   width=0.4)
ax.bar_label(ax.containers[0])
ax.tick_params(axis='x', rotation=0)
ax.set_title('State that Having TOP 10 Higest Disputes')
ax.set_ylabel('No of Consumer Disputes')
plt.show()


# **From the figure above, we can tell California has the maximum disputes**

# Plot bar graph for the total no of disputes by Submitted Via diffrent source

# In[65]:


sns.set_theme(rc={'figure.figsize':(17, 8)})        # set the context with a specific size
ax = sns.countplot(consumer_complaints_train_df,
                   x='Submitted via',
                   hue='Consumer disputed?',
                   order=disputed_cons['Submitted via'].value_counts().sort_values(ascending=False).index,
                   width=0.4)
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.tick_params(axis='x', rotation=0)
ax.set_title('The Total No of Disputes Submitted by Different Source')
plt.show()


# Plot bar graph for the total no of disputes where Company's Response to the Complaints

# In[67]:


# Complaints: Having or Not having Disputes
# Disputes: Only Conusmer disputed
# draw barplots to display issues that having TOP 10 highst disputes
sns.set_theme(rc={'figure.figsize':(17, 8)})        # set the context with a specific size
ax = sns.countplot(consumer_complaints_train_df,
                   x='Company response to consumer',
                   hue='Consumer disputed?',
                   order=consumer_complaints_train_df['Company response to consumer'].value_counts().sort_values(ascending=False).index)
ax.tick_params(axis='x', rotation=0)
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.set_title('The Total No of Disputes where Company\'s Response to the Complaints')
ax.set_xlabel('Where Company Gives Response')
ax.set_ylabel('Total Disputes')
plt.show()


# Plot bar graph for the total no of disputes where Company's Response Leading to Disputes

# In[69]:


sns.set_theme(rc={'figure.figsize':(17, 8)})        # set the context with a specific size
ax = sns.countplot(disputed_cons,
                   x='Company response to consumer',
                   order=disputed_cons['Company response to consumer'].value_counts().sort_values(ascending=False).index,
                   width=0.4)
ax.bar_label(ax.containers[0])
ax.tick_params(axis='x', rotation=0)
ax.set_title('The Total No of Disputes where Company\'s Response to the Complaints')
ax.set_xlabel('No of Consumer Disputes')
ax.set_ylabel('Total Disputes')
plt.show()


# **From the figure above, we can tell 'Closed with explanation' is the main cause leading to Disputes**

# Plot bar graph for the total no of disputes Whether there are Disputes Instead of Timely Response

# In[72]:


sns.set_theme(rc={'figure.figsize':(6, 5)})        # set the context with a specific size
ax = sns.countplot(disputed_cons,
                   x='Timely response?',
                   width=0.4)
ax.bar_label(ax.containers[0])
ax.tick_params(axis='x', rotation=0)
ax.set_title('Timely Reponse in All Disputes')
ax.set_xlabel('Timely Reponse?')
ax.set_ylabel('Count')
plt.show()


# In[73]:


non_timely_response_in_disputes = 1229 / (1229 + 74943)
np.round(non_timely_response_in_disputes * 100, 3)


# **From the figure above, we can tell among all disputes, there are few having Non-timely response ones indeed, accounting for
# 1.61% among all disputes**

# Plot bar graph for the total no of disputes over Year Wise Complaints

# In[76]:


sns.set_theme(rc={'figure.figsize':(17, 8)})        # set the context with a specific size
ax = sns.countplot(consumer_complaints_train_df,
                   x='Year_received',
                   width=0.4)
ax.bar_label(ax.containers[0])
ax.tick_params(axis='x', rotation=0)
ax.set_title('Total No of Complaints by Year')
ax.set_xlabel('Year_received')
ax.set_ylabel('No of Complaints')
plt.show()


# Plot bar graph for the total no of disputes over Year Wise Disputes

# In[78]:


sns.set_theme(rc={'figure.figsize':(17, 8)})        # set the context with a specific size
ax = sns.countplot(disputed_cons,
                   x='Year_received',
                   width=0.4)
ax.bar_label(ax.containers[0])
ax.tick_params(axis='x', rotation=0)
ax.set_title('Total No of Disputes by Year')
ax.set_xlabel('Year')
ax.set_ylabel('No of Disputes')
plt.show()


# Plot  bar graph for the top companies with highest complaints

# In[80]:


sns.set_theme(rc={'figure.figsize':(17, 8)})        # set the context with a specific size
ax = sns.countplot(consumer_complaints_train_df,
                   x='Company',
                   order=consumer_complaints_train_df['Company'].value_counts()[0:10].sort_values(ascending=False).index)
ax.bar_label(ax.containers[0])
ax.tick_params(axis='x', rotation=30)
ax.set_title('Top 10 Companies with Highest Complaints')
ax.set_xlabel('Company')
ax.set_ylabel('No of Complaints')
plt.show()


# "Days Held" Column Analysis(describe)

# In[82]:


pd.DataFrame(consumer_complaints_train_df['Days held'].describe())


# In[83]:


pd.DataFrame(consumer_complaints_test_df['Days held'].describe())


# Convert Negative Days Held to Zero(it is the time taken by authority can't be negative)

# In[85]:


# for training set
train_filter = consumer_complaints_train_df['Days held'] < 0
consumer_complaints_train_df.loc[train_filter, 'Days held'] = 0

# for testing set
test_filter = consumer_complaints_test_df['Days held'] < 0
consumer_complaints_test_df.loc[test_filter, 'Days held'] = 0


# In[86]:


consumer_complaints_train_df['Days held'].describe()


# In[87]:


df_train, df_test = consumer_complaints_train_df, consumer_complaints_test_df


# Text pre-processing

# In[89]:


# Tokenize text data
tokenized_data_train = df_train['Issue'].apply(lambda x: wordpunct_tokenize(x.lower()))
tokenized_data_test = df_test['Issue'].apply(lambda x: wordpunct_tokenize(x.lower()))

# Remove punctuation
def remove_punctuation(text):
    return [w for w in text if w.isalpha()]
no_punctuation_data_train = tokenized_data_train.apply(lambda x: remove_punctuation(x))
no_punctuation_data_test = tokenized_data_test.apply(lambda x: remove_punctuation(x))

# Remove stop words
stop_words = stopwords.words('english')
filtered_sentence_train = no_punctuation_data_train.apply(lambda x: [w for w in x if not w in stop_words])
filtered_sentence_test = no_punctuation_data_test.apply(lambda x: [w for w in x if not w in stop_words])

# Lemmatize text
lemmatized_data_train = filtered_sentence_train.apply(lambda x: [WordNetLemmatizer().lemmatize(w, pos='v') for w in x])
lemmatized_data_test = filtered_sentence_test.apply(lambda x: [WordNetLemmatizer().lemmatize(w, pos='v') for w in x])

# Stem text
stemmed_data_train = lemmatized_data_train.apply(lambda x: [PorterStemmer().stem(w) for w in x])
stemmed_data_test = lemmatized_data_test.apply(lambda x: [PorterStemmer().stem(w) for w in x])

# Convert words to sentences
clean_data_train = stemmed_data_train.apply(lambda x: ' '.join(x))
clean_data_test = stemmed_data_test.apply(lambda x: ' '.join(x))


# In[90]:


df_train['Issues_cleaned'] = clean_data_train
df_test['Issues_cleaned'] = clean_data_test
df_train = df_train.drop('Issue', axis = 1)
df_test = df_test.drop('Issue', axis = 1)


# Drop Unnecessary Columns for the Model Building<br>
# like:'Company', 'State', 'Year_Received', 'Days_held'

# In[92]:


df_train.drop(columns=['Company', 'State', 'Year_received', 'Days held'], axis=1, inplace=True)
df_test.drop(columns=['Company', 'State', 'Year_received', 'Days held'], axis=1, inplace=True)


# Change Consumer Disputed Column to 0 and 1(yes to 1, and no to 0)

# In[94]:


con_dis_to_rep = dict(zip(['Yes', 'No'], [1, 0]))
df_train.replace({'Consumer disputed?': con_dis_to_rep}, inplace=True)


# In[95]:


df_train.head(2)


# Create Dummy Variables for catagorical features
# like: 'Product', 'Submitted via', 'Company response to consumer', 'Timely response?'

# In[97]:


dummy_cols = ['Product', 'Submitted via', 'Company response to consumer', 'Timely response?']

df_train_dummies = pd.get_dummies(df_train[dummy_cols], prefix_sep='_', drop_first=True)
df_test_dummies = pd.get_dummies(df_test[dummy_cols], prefix_sep='_', drop_first=True)
df_train_dummies.head(2)


# Concate Dummy Variables and Drop the Original Columns

# In[99]:


# drop original columns
df_train.drop(dummy_cols, axis=1, inplace=True)
df_test.drop(dummy_cols, axis=1, inplace=True)

# concate dummy variables with the original dataframe
df_train = pd.concat([df_train, df_train_dummies], axis=1)
df_test = pd.concat([df_test, df_test_dummies], axis=1)

df_train.head(2)


# In[100]:


df_train.info()


# Calculating TF-IDF

# In[102]:


tf = TfidfVectorizer()

issues_cleaned_train = tf.fit_transform(df_train['Issues_cleaned']).toarray()
issues_cleaned_test = tf.fit_transform(df_test['Issues_cleaned']).toarray()

tf_columns_train = []
tf_columns_test = []

for i in range(issues_cleaned_train.shape[1]):
    tf_columns_train.append('Feature' + str(i+1))
for i in range(issues_cleaned_test.shape[1]):
    tf_columns_test.append('Feature' + str(i+1))
    
issues_train = pd.DataFrame(issues_cleaned_train, columns = tf_columns_train)
issues_test = pd.DataFrame(issues_cleaned_test, columns = tf_columns_test)

weights = pd.DataFrame(tf.idf_, columns = ['Idf_weights']).sort_values(by = 'Idf_weights', ascending = False)
weights.head()


# In[103]:


tf_columns_train[-1]


# In[104]:


tf_columns_test[-1]


# Replacing Issues_cleaned by Vectorized Issues

# In[106]:


df_train = df_train.drop('Issues_cleaned', axis = 1)
df_test = df_test.drop('Issues_cleaned', axis = 1)
df_train = pd.concat([df_train, issues_train], axis = 1)
df_test = pd.concat([df_test, issues_test], axis = 1)
Feature168 = [0] * 119606
df_test['Feature168'] = Feature168


# In[107]:


df_train.shape


# In[108]:


df_test.shape


# observe train and test datasets

# In[110]:


df_train.head()


# In[111]:


df_test.head()


# Observe Shape of new Train and Test Datasets

# In[113]:


# shape of new train datasets
df_train.shape


# In[114]:


df_test.shape


# Scaling the Data Sets (note:discard dependent variable before doing standardization)

# In[116]:


std = StandardScaler()

df_train_scaled = pd.DataFrame(std.fit_transform(df_train.drop('Consumer disputed?', axis=1)), columns=df_test.columns)
df_test_scaled = pd.DataFrame(std.fit_transform(df_test), columns=df_test.columns)


# In[117]:


df_train_scaled.head(3)


# Do feature selection with help of PCA

# In[119]:


# Projection of PCA
principal_columns = []
list(map(lambda index: principal_columns.append('PC' + str(index + 1)), range(df_train_scaled.shape[1])))

pca = PCA()
principal_components = pca.fit_transform(df_train_scaled)
principalDf = pd.DataFrame(data=principal_components, columns=principal_columns)
principalDf.head(2)


# In[120]:


# Calculate explained ratio, and then display the variance and importance then
explained_ratio_train_df = pd.DataFrame(pca.explained_variance_ratio_, columns=['Explained Ratio']).sort_values(by='Explained Ratio', ascending=False)
importance = []
list(map(lambda index: importance.append(explained_ratio_train_df['Explained Ratio'].head(index+1).sum()), range(explained_ratio_train_df.shape[0])))
explained_ratio_train_summary = pd.DataFrame({'Variable': principal_columns, 'Importance': importance})
explained_ratio_train_summary.head()


# Select top features which are covering 80% of the information
# (n=53),
# <br>store this data into new dataframe,

# In[122]:


# Select top features which are covering 80% of the information (n=53)
pca_train_model = PCA(n_components=53)
pca_test_model = PCA(n_components=53)

trained_pca = pca_train_model.fit_transform(df_train_scaled)
tested_pca = pca_test_model.fit_transform(df_test_scaled)

# store the data into new dataframe
trained_pca_df = pd.DataFrame(data=trained_pca, columns=[f'PC{i+1}' for i in range(trained_pca.shape[1])])
trained_pca_df.head()


# In[123]:


tested_pca_df = pd.DataFrame(data=tested_pca, columns=[f'PC{i+1}' for i in range(tested_pca.shape[1])])
tested_pca_df.head()


# Split the Data Sets Into X and Y by dependent and independent variables (data selected by PCA)
# 

# In[125]:


X, y = trained_pca_df, df_train['Consumer disputed?']


# Split data into Train and Test datasets
# (for test data use test excel file data)

# In[127]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42)
X_test = tested_pca


# Shapes of the datasets

# In[129]:


X_train.shape, X_val.shape, y_train.shape, y_val.shape, X_test.shape


# **Model building**
# Build given models and mesure their test and validation accuracy
# build given models:
# 1. LogisticRegression
# 2. DecisionTreeClassifier
# 3. RandomForestClassifier
# 4. AdaBoostClassifier
# 5. GradientBoostingClassifier
# 6. KNeighborsClassifier
# 7. XGBClassifier

# In[131]:


models = [LogisticRegression(),
          DecisionTreeClassifier(),
          RandomForestClassifier(),
          AdaBoostClassifier(),
          GradientBoostingClassifier(),
          KNeighborsClassifier(),
          XGBClassifier()]
model_names = [
    'LogisticRegression',
    'DecisionTreeClassifier',
    'RandomForestClassifier',
    'AdaBoostClassifier',
    'GradientBoostingClassifier',
    'KNeighborsClassifier',
    'XGBClassifier'
]
acc_score_train = []
acc_score_val = []

def accuracy_collection(model):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    acc_score_train.append(accuracy_score(y_train, y_pred_train))
    acc_score_val.append(accuracy_score(y_val, y_pred_val))

list(map(lambda model: accuracy_collection(model), models))


# In[132]:


train_val_acc_dict = {'Modelling Algorithm': model_names,
                      'Training Accuracy': acc_score_train,
                      'Validation Accuracy': acc_score_val}
train_val_acc_df = pd.DataFrame(train_val_acc_dict).sort_values(by='Validation Accuracy', ascending=False)
train_val_acc_df


# In[133]:


# Plot all the accuracies in the diagram
sns.set_theme(rc={'figure.figsize':(17, 8)})        # set the context with a specific size
ax = sns.barplot(x='Modelling Algorithm',
                 y='Validation Accuracy',
                 data=train_val_acc_df)
ax.bar_label(ax.containers[0])
ax.tick_params(axis='x', rotation=30)
ax.set_title('Classifier Accuracy in Validation Set (Sorting By Validation Accuray from High to Low)')
ax.set_xlabel('Classifiers')
ax.set_ylabel('Validation Accuracy')
plt.show()


# Final Model and Prediction for test data file

# In[135]:


# From the results shown above, LogisticRegression and AdaboostClassifier is the best classifier
best_classfier = LogisticRegression()
best_classfier.fit(X_train, y_train)

# make prediction on the testing dataset
consumer_complaints_test_df['Consumer disputed?'] = best_classfier.predict(X_test)


# Export Predictions to CSV

# In[137]:


# save the testing dataset with prediction results together to the file
consumer_complaints_test_df.to_csv('test_data_with_prediction.csv', index=False)


# In[138]:


consumer_complaints_test_df['Consumer disputed?'].value_counts()

