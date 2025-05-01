import pandas as pd
from PyQt5.QtWidgets import (QWidget, QLineEdit, QTableWidget,
                             QPushButton, QLabel, QGridLayout, QMessageBox, 
                             QTableWidgetItem)
from recommendation_algorithms.content_based_recommender import content_based_recommender

class ContentAppPyQt(QWidget):
    def __init__(self, movies_df):
        super().__init__()

        self.db = pd.DataFrame(columns=['Sl.No', 'Movie Title'])
        self.movies_df = movies_df
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Content-based Recommender System")

        layout = QGridLayout()

        # Movie title input
        layout.addWidget(QLabel("Movie Title:"), 0, 0)
        self.movie_title_input = QLineEdit(self)
        layout.addWidget(self.movie_title_input, 0, 1)

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
        self.output_table.setColumnCount(2)  # Number of columns
        self.output_table.setHorizontalHeaderLabels(
            ['Sl.No', 'Movie Title']
        )
        layout.addWidget(self.output_table, 4, 0, 1, 2)

        self.setLayout(layout)

    def on_button_clicked(self):
        movie_title = self.movie_title_input.text()
        num_recommendations = self.num_recommendations_input.text()

        if not movie_title or not num_recommendations:
            QMessageBox.warning(self, "Error", "Please fill out all fields before recommending!")
            return

        try:
            num_recommendations = int(num_recommendations)
        except ValueError:
            QMessageBox.warning(self, "Error", "Number of recommendations must be integers!")
            return

        recommendations = content_based_recommender(self.movies_df, movie_title, num_recommendations)
        self.db = pd.concat([self.db, recommendations], ignore_index=True)

        self.populate_table(recommendations)

    def populate_table(self, data):
        """Populate QTableWidget with DataFrame data."""
        self.output_table.setRowCount(len(data))  # Set number of rows

        for i, row in data.iterrows():
            self.output_table.setItem(i, 0, QTableWidgetItem(str(row['Sl.No'])))
            self.output_table.setItem(i, 1, QTableWidgetItem(row['Movie Title']))