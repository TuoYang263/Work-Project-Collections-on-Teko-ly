import sys
from user_interface.main_app import MainApp
from support_modules.data_preprocessing import load_data, preprocess_data
from recommendation_algorithms.content_based_recommender import content_based_recommender
from recommendation_algorithms.collaborative_recommender import collaborative_recommender
from recommendation_algorithms.popularity_recommender import popularity_recommender
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    # Load and preprocess data
    movies_df, ratings_df = load_data()
    movies_df = preprocess_data(movies_df)
    
    app = QApplication(sys.argv)
    
    # Main application window
    main_window = MainApp(movies_df, ratings_df)
    main_window.resize(800, 600)
    main_window.show()
    
    sys.exit(app.exec_())
    
    
