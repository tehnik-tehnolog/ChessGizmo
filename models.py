import pickle
import pandas as pd
import numpy as np


class ChessModel:
    def __init__(self, chess_games_info:pd.DataFrame, game_type:str, clean_outliers:bool=True):
        self.chess_games_info = chess_games_info
        self.game_type = game_type
        self.clean_outliers = clean_outliers
        self.cgi_no_out_norm = None
        self.outcome_column = None
        self.default_model_path = fr'./models/{self.game_type}'

        self.clean_and_scale_df()

    def clean_and_scale_df(self, model_path: str = None):
        if model_path is None:
            model_path = self.default_model_path
        with open(model_path + r'/data_transformer.pkl', 'rb') as data_transformer_file:
            data_transformer = pickle.load(data_transformer_file)

        if self.clean_outliers:
            with open(model_path + r'/clf.pkl', 'rb') as clf_file:
                clf = pickle.load(clf_file)
            outliers = clf.predict(self.chess_games_info)
        else:
            outliers = np.zeros(self.chess_games_info.shape[0])
        df_no_out = self.chess_games_info[outliers == 0]
        self.outcome_column = df_no_out['outcome']
        self.cgi_no_out_norm = pd.DataFrame(data_transformer.fit_transform(df_no_out))

    def rating_predict(self, model_path: str = None):
        if model_path is None:
            model_path = self.default_model_path
        with open(model_path + r'/catboost_regression.pkl', 'rb') as cbr_file:
            cat_boost_regressor = pickle.load(cbr_file)
        rating_pred = cat_boost_regressor.predict(self.cgi_no_out_norm)
        return rating_pred

    def master_predict(self, model_path: str = None):
        if model_path is None:
            model_path = self.default_model_path
        with open(model_path + r'/knn_classificator.pkl', 'rb') as knn_file:
            knn_classificator = pickle.load(knn_file)
        master_pred = knn_classificator.predict(self.cgi_no_out_norm.iloc[:, 1:9])
        return master_pred

    def get_rating_dict(self):
        df = pd.DataFrame()
        df['outcome'] = self.outcome_column
        df['rating_pred'] = self.rating_predict()
        rating_dict = df.groupby('outcome')['rating_pred'].mean().round().astype(int).to_dict()
        for key in [1.0, 0.0, 0.5]:
            rating_dict.setdefault(key, None)
        return rating_dict

    def get_game_style(self, boundary_share=0.05):
        master_arr = self.master_predict()
        masters, counts = np.unique(master_arr, return_counts=True)
        counts_share = counts / sum(counts)

        style_dict = {m: s for m, s in sorted(zip(masters, counts_share), key=lambda x: x[1], reverse=True) if
                      s >= boundary_share}

        return style_dict