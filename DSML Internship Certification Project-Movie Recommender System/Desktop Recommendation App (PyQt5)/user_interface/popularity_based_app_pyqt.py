import pandas as pd
from PyQt5.QtWidgets import (QWidget,QLineEdit, QTableWidget,
                             QPushButton, QLabel, QGridLayout, QMessageBox, 
                             QTableWidgetItem)
from recommendation_algorithms.popularity_recommender import popularity_recommender

class PopularityAppPyQt(QWidget):
    def __init__(self, movies_df, ratings_df):
        super().__init__()
        self.db = pd.DataFrame(columns=['S.No', 'Movie Title', 'Average Movie Rating', 'Num Reviews'])
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Popularity-based Recommender System")

        layout = QGridLayout()

        # Genre input
        layout.addWidget(QLabel("Genre:"), 0, 0)
        self.genre_input = QLineEdit(self)
        layout.addWidget(self.genre_input, 0, 1)

        # Minimum reviews input
        layout.addWidget(QLabel("Minimum reviews threshold:"), 1, 0)
        self.min_reviews_input = QLineEdit(self)
        layout.addWidget(self.min_reviews_input, 1, 1)

        # Number of recommendations input
        layout.addWidget(QLabel("Number of recommendations:"), 2, 0)
        self.num_recommendations_input = QLineEdit(self)
        layout.addWidget(self.num_recommendations_input, 2, 1)

        # Recommend button
        self.recommend_button = QPushButton("Recommend Movies", self)
        self.recommend_button.clicked.connect(self.on_button_clicked)
        layout.addWidget(self.recommend_button, 3, 0, 1, 2)

        # Output table
        self.output_table = QTableWidget(self)
        self.output_table.setColumnCount(4)  # Number of columns
        self.output_table.setHorizontalHeaderLabels(
            ['S.No', 'Movie Title', 'Average Movie Rating', 'Num Reviews']
        )
        layout.addWidget(self.output_table, 4, 0, 1, 2)

        self.setLayout(layout)

    def on_button_clicked(self):
        genre = self.genre_input.text()
        min_reviews = self.min_reviews_input.text()
        num_recommendations = self.num_recommendations_input.text()

        if not genre or not min_reviews or not num_recommendations:
            QMessageBox.warning(self, "Error", "Please fill out all fields before recommending!")
            return

        try:
            min_reviews = int(min_reviews)
            num_recommendations = int(num_recommendations)
        except ValueError:
            QMessageBox.warning(self, "Error", "Minimum reviews and Number of recommendations must be integers!")
            return
            
        recommendations = popularity_recommender(self.movies_df, self.ratings_df, genre, min_reviews, num_recommendations)
        self.db = pd.concat([self.db, recommendations], ignore_index=True)

        self.populate_table(recommendations)

    def populate_table(self, data):
        """Populate QTableWidget with DataFrame data."""
        self.output_table.setRowCount(len(data))  # Set number of rows

        for i, row in data.iterrows():
            self.output_table.setItem(i, 0, QTableWidgetItem(str(row['S.No'])))
            self.output_table.setItem(i, 1, QTableWidgetItem(row['Movie Title']))
            self.output_table.setItem(i, 2, QTableWidgetItem(str(row['Average Movie Rating'])))
            self.output_table.setItem(i, 3, QTableWidgetItem(str(row['Num Reviews'])))