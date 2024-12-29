
# Movie Recommender System Project

## Overview
This project implements a comprehensive **Movie Recommender System**, developed as part of a DSML (Data Science and Machine Learning) Internship Certification. The system integrates multiple recommendation algorithms and demonstrates its capabilities through both desktop and web applications.

The recommender system includes the following approaches:
- **Collaborative Filtering**
- **Content-Based Filtering**
- **Popularity-Based Filtering**

It utilizes PyQt5 for the desktop application and Flask for the web application, providing an end-to-end solution for delivering movie recommendations.

---

## File Structure

```
DSML Internship Certification Project/
├── DSML Certification Project-Movie Recommender System.pdf
├── README.md
├── requirements.txt
├── data/
│   ├── movies.csv
│   ├── ratings.csv
├── Desktop Recommendation App (PyQt5)/
│   ├── pyqt5_app_main.py
│   ├── Readme.txt
│   ├── recommendation_algorithms/
│   │   ├── collaborative_recommender.py
│   │   ├── content_based_recommender.py
│   │   ├── popularity_recommender.py
│   ├── support_modules/
│   │   ├── data_preprocessing.py
│   ├── user_interface/
│   │   ├── collaborative_filtering_app_pyqt.py
│   │   ├── content_based_app_pyqt.py
│   │   ├── main_app.py
│   │   ├── popularity_based_app_pyqt.py
├── Flask Recommendation App/
│   ├── app.py
│   ├── readme.txt
│   ├── program_running_results/
│   │   ├── collaborative-filtering recommendation results.png
│   │   ├── collaborative-filtering recommendation user interface.png
│   │   ├── content-based recommendation results.png
│   │   ├── content-based recommendation user interface.png
│   │   ├── popularity-based recommendation interface.png
│   │   ├── popularity-based recommendation results.png
│   ├── recommendation_algorithms/
│   │   ├── collaborative_recommender.py
│   │   ├── content_based_recommender.py
│   │   ├── popularity_recommender.py
│   ├── support_modules/
│   │   ├── data_preprocessing.py
│   ├── templates/
│       ├── index.html
│       ├── results.html
├── notebooks/
│   ├── movie_recommender_system_notebook.ipynb
```

---

## Project Components

### 1. Data
- **movies.csv**: Contains metadata about movies.
- **ratings.csv**: Includes user ratings for movies.

### 2. Desktop Application (PyQt5)
A standalone desktop application that provides movie recommendations based on the selected algorithm. The application uses modular designs:
- **Recommendation Algorithms**:
  - Collaborative Filtering
  - Content-Based Filtering
  - Popularity-Based Filtering
- **Support Modules**: Handles data preprocessing and other utilities.
- **User Interface**: PyQt5-based GUI components.

### 3. Web Application (Flask)
A Flask-based web application that enables users to get movie recommendations through a browser interface. It consists of:
- Flask backend (`app.py`)
- Recommendation Algorithms (modular structure for flexibility)
- Predefined HTML templates for user interaction.

### 4. Jupyter Notebook
An exploratory notebook (`movie_recommender_system_notebook.ipynb`) detailing data preprocessing, exploratory data analysis (EDA), and the development of recommendation algorithms.

---

## Installation

### Prerequisites
- Python 3.10 or higher
- Libraries specified in `requirements.txt`

### Steps
1. Clone the repository or download the zip file.
2. Navigate to the project directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Desktop Application
1. Navigate to the `Desktop Recommendation App (PyQt5)` folder.
2. Run `pyqt5_app_main.py` using Python 3.10 or higher.
3. Choose a recommendation algorithm from the GUI to receive movie recommendations.

### Web Application
1. Navigate to the `Flask Recommendation App` folder.
2. Run `app.py` using Python 3.10 or higher.
3. Open the local server URL in a browser to access the app.

### Jupyter Notebook
1. Open `movie_recommender_system_notebook.ipynb` in Jupyter Notebook or any compatible IDE.
2. Execute the cells sequentially to explore the data and algorithms.

---

## Key Features
- Modular codebase with separate folders for algorithms, preprocessing, and user interfaces.
- Dual-platform deployment: Desktop (PyQt5) and Web (Flask).
- Clear visualizations and results for each recommendation approach.
- Comprehensive documentation for both desktop and web apps.

---

## Requirements
- Python 3.10 or higher
- Required Python Libraries: `pandas`, `numpy`, `scikit-learn`, `PyQt5`, `Flask`, `matplotlib`, `wordcloud`

---

## Author
Developed as part of the **DSML Internship Certification Program**.

---

## License
This project is licensed under the MIT License.

---

## Acknowledgments
Special thanks to the DSML Internship Certification team for guidance and resources.
