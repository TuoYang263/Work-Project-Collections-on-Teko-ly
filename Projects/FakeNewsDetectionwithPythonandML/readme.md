# Fake News Detection with Python and Machine Learning
## _By Tuo Yang_

## Objectives
The goal of this project is to make a difference between real and fake news with python. Using sklearn, a TfidfVectorizer on the dataset will be built. Then, a PassiveAggressiveClassifier will be initialized and used to fit the model. Finally, the accuracy score and the confusion matrix will evaluate how well this model is.

## Algorithms Implemented to Reach Objectives
Two algorithms will be used right here: **TfidfVectorizer and PassiveAggerssiveClassifier**
1. The TfidfVectorizer converts a collection of raw documents into a matrix of TF-IDF features
2. Passive Aggressive algorithms are online learning algorithms. Such an algorithm remains passive for a correct classification outcome, and turns aggressive in the event of a miscalculation, updating, and <br>adjusting. Unlike most other algorithms, it does not converge.Its purpose is to make updates that correct the loss, causing very little change in the norm of the weight vector
 
## Project File Architecuture
- **Datasets** : It is a folder which contains:
    - **news.csv**: A dataset contains 6335 records used for fake news ML model's training and testing
- **Evaluation Results** - It is a folder which contains:
    - **model_evaluation_results_part1.JPG**: Evaluation results of metrics of accuracy, confusion matrix, precision, and recall on testing set.
    - **model_evaluation_results_part2.JPG**: Evaluation results of metrics of f1-score and ROC AUC score (Area under ROC curve)  on testing set.
    - **model_evaluation_results_part2.JPG**: Evaluation results displayed in AUC ROC curve) on testing set.
- **fake_news_detection.ipynb**: The Jupyter Notebook used for training and testing ML models for fake news detection.