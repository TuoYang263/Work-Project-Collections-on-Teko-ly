import pandas as pd
from PyQt5.QtWidgets import (QWidget, QLineEdit, QTableWidget,
                             QPushButton, QLabel, QGridLayout, QMessageBox, 
                             QTableWidgetItem)
from recommendation_algorithms.collaborative_recommender import collaborative_recommender

class CollaborativeAppPyQt(QWidget):
    def __init__(self, movies_df, ratings_df):
        super().__init__()

        self.db = pd.DataFrame(columns=['S.No', 'Movie Title'])
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Collaborative Filtering Recommender System")

        layout = QGridLayout()

        # User ID input
        layout.addWidget(QLabel("UserID:"), 0, 0)
        self.user_id_input = QLineEdit(self)
        layout.addWidget(self.user_id_input, 0, 1)

        # Number of recommendations input
        layout.addWidget(QLabel("Number of recommendations:"), 1, 0)
        self.num_recommendations_input = QLineEdit(self)
        layout.addWidget(self.num_recommendations_input, 1, 1)

        # Threshold for similar users input
        layout.addWidget(QLabel("Threshold for similar users:"), 2, 0)
        self.thresh_similar_user_input = QLineEdit(self)
        layout.addWidget(self.thresh_similar_user_input, 2, 1)

        # Recommend button
        self.recommend_button = QPushButton("Recommend Movies", self)
        self.recommend_button.clicked.connect(self.on_button_clicked)
        layout.addWidget(self.recommend_button, 3, 0, 1, 2)

        # Output table
        self.output_table = QTableWidget(self)
        self.output_table.setColumnCount(2)  # Number of columns
        self.output_table.setHorizontalHeaderLabels(
            ['S.No', 'Movie Title']
        )
        layout.addWidget(self.output_table, 4, 0, 1, 2)

        self.setLayout(layout)

    def on_button_clicked(self):
        user_id = self.user_id_input.text()
        num_recommendations = self.num_recommendations_input.text()
        thresh_for_similar_users = self.thresh_similar_user_input.text()

        if not user_id or not num_recommendations or not thresh_for_similar_users:
            QMessageBox.warning(self, "Error", "Please fill out all fields before recommending!")
            return

        try:
            user_id = int(user_id)
            thresh_for_similar_users = int(thresh_for_similar_users)
            num_recommendations = int(num_recommendations)
        except ValueError:
            QMessageBox.warning(self, "Error", "User ID, Number of recommendations, and threshold fo similar users must be integers!")
            return

        recommendations = collaborative_recommender(self.movies_df, self.ratings_df, user_id, num_recommendations, thresh_for_similar_users)
        self.db = pd.concat([self.db, recommendations], ignore_index=True)

        self.populate_table(recommendations)

    def populate_table(self, data):
        """Populate QTableWidget with DataFrame data."""
        self.output_table.setRowCount(len(data))  # Set number of rows

        for i, row in data.iterrows():
            self.output_table.setItem(i, 0, QTableWidgetItem(str(row['S.No'])))
            self.output_table.setItem(i, 1, QTableWidgetItem(row['Movie Title']))