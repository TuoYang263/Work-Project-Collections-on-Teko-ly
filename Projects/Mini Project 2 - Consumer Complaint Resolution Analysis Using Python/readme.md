# Consumer Complaint Resolution Analysis Using Python

## Data Science and Machine Learning Internship Program

### Mini Project 2: Consumer Complaint Resolution

#### Project Overview

This project involves analyzing consumer complaint data to understand customer sentiment and resolve issues more effectively. When customers are dissatisfied, they often raise complaints. Although businesses strive to resolve these complaints, not all customers are satisfied.

By analyzing the complaint data using various machine learning algorithms, we aim to classify customers accurately and predict whether a customer will dispute their complaint.

#### Objective

The primary objective is to use Python libraries to analyze the data and build models to predict whether a customer will dispute a complaint. Key libraries include:

- **Pandas** for data manipulation
- **Seaborn** and **Matplotlib** for data visualization and exploratory data analysis (EDA)
- **Scikit-Learn** for model building and performance evaluation

The best-performing model will be used to predict the outcome for the test data and provide actionable insights.

### Dataset Description

The dataset contains the following information:

- **Dispute**: Target variable indicating whether a customer has disputed (Yes/No).
- **Date Received**: The date the complaint was received.
- **Product**: Product categories (e.g., credit cards, loans).
- **Sub-product**: Specific options within the product (e.g., insurance, mortgage).
- **Issue**: Details of the customer complaint.
- **Company Public Response**: How the company responded to the complaint.
- **Company**: Name of the company.
- **State**: Customer's state of residence.
- **ZIP Code**: Customer’s ZIP code.
- **Submitted Via**: Platform used to register complaints (e.g., web, phone).
- **Date Sent to Company**: The date the complaint was registered.
- **Timely Response?**: Whether the company responded in a timely manner (Yes/No).
- **Consumer Disputed?**: Target variable indicating dispute status (Yes/No).
- **Complaint ID**: Unique identifier for each complaint.

### Tasks to Perform

1. **Data Preparation and Cleaning:**
   - Read data from the provided Excel files.
   - Verify data types for both training and test datasets.
   - Analyze missing values and drop columns where more than 25% of the data is missing.
   - Extract day, month, and year from the "Date Received" column.
   - Calculate the number of days the complaint was with the company and create a "Days Held" field.
   - Drop unnecessary fields: "Date Received," "Date Sent to Company," "ZIP Code," "Complaint ID."
   - Impute missing values in the "State" column using the mode.
   - Create a "Week Received" field using the calculated days.
   - Store disputed customer data for further analysis.

2. **Data Visualization:**
   - Plot bar graphs using Seaborn for the following:
     - Total number of consumer disputes.
     - Disputes by product type.
     - Top issues causing disputes.
     - Disputes by state.
     - Disputes by submission method.
     - Company responses and their impact on disputes.
     - Disputes despite timely responses.
     - Year-wise complaint and dispute trends.
     - Companies with the highest number of complaints.

3. **Data Preprocessing for Model Building:**
   - Convert negative "Days Held" values to zero.
   - Drop unnecessary columns: 'Company', 'State', 'Year Received', 'Days Held'.
   - Encode the "Consumer Disputed" column as 0 (No) and 1 (Yes).
   - Create dummy variables for categorical features (e.g., 'Product', 'Submitted Via', 'Company Response to Consumer', 'Timely Response?').
   - Scale datasets and apply Principal Component Analysis (PCA) to retain up to 80% of variance.

4. **Model Development and Evaluation:**
   - Split the dataset into features (X) and the target variable (Y).
   - Build and evaluate the following models:
     - Logistic Regression
     - Decision Tree Classifier
     - Random Forest Classifier
     - AdaBoost Classifier
     - Gradient Boosting Classifier
     - K-Nearest Neighbors (KNN)
     - XGBoost Classifier
   - Select the model with the highest accuracy to predict outcomes for the test file.

[Datasets link](https://drive.google.com/drive/folders/14E-ixaMFUU3bskNm4gC3iTL00Ei9avFw?usp=sharing)

[Prediction results](https://drive.google.com/drive/folders/16G7fXKtVto5PAgixZf2XGYggACkRanBW?usp=sharing)