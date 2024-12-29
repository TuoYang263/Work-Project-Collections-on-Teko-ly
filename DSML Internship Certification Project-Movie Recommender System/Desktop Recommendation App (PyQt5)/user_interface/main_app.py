from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QMainWindow, QStackedWidget)
from user_interface.popularity_based_app_pyqt import PopularityAppPyQt
from user_interface.content_based_app_pyqt import ContentAppPyQt
from user_interface.collaborative_filtering_app_pyqt import CollaborativeAppPyQt

class MainApp(QMainWindow):
    def __init__(self, movies_df, ratings_df):
        super().__init__()
        self.setWindowTitle("Recommendation System")

        # Central widget to hold the stacked layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout for navigation
        self.main_layout = QVBoxLayout()

        # Navigation buttons
        self.popularity_button = QPushButton("Popularity-Based Recommender")
        self.popularity_button.clicked.connect(self.show_popularity)
        self.main_layout.addWidget(self.popularity_button)

        self.content_button = QPushButton("Content-Based Recommender")
        self.content_button.clicked.connect(self.show_content)
        self.main_layout.addWidget(self.content_button)

        self.collaborative_button = QPushButton("Collaborative Filtering Recommender")
        self.collaborative_button.clicked.connect(self.show_collaborative)
        self.main_layout.addWidget(self.collaborative_button)

        # Stacked widget to switch between UIs
        self.stacked_widget = QStackedWidget()

        # Add the individual recommendation systems as pages
        self.popularity_ui = PopularityAppPyQt(movies_df, ratings_df)
        self.content_ui = ContentAppPyQt(movies_df)
        self.collaborative_ui = CollaborativeAppPyQt(movies_df, ratings_df)

        self.stacked_widget.addWidget(self.popularity_ui)
        self.stacked_widget.addWidget(self.content_ui)
        self.stacked_widget.addWidget(self.collaborative_ui)

        self.main_layout.addWidget(self.stacked_widget)

        self.central_widget.setLayout(self.main_layout)

    def show_popularity(self):
        self.stacked_widget.setCurrentWidget(self.popularity_ui)

    def show_content(self):
        self.stacked_widget.setCurrentWidget(self.content_ui)

    def show_collaborative(self):
        self.stacked_widget.setCurrentWidget(self.collaborative_ui)