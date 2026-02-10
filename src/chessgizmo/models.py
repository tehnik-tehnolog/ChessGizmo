import pickle
import pandas as pd
import numpy as np
from importlib import resources


class ChessModel:
    def __init__(self, chess_games_info:pd.DataFrame, game_type:str, clean_outliers:bool=True):
        self.chess_games_info = chess_games_info
        self.game_type = game_type
        self.clean_outliers = clean_outliers
        self.cgi_no_out_norm = None
        self.outcome_column = None
        self.default_model_path = f'chessgizmo.data.models.{self.game_type}'

        self.clean_and_scale_df()

    def _load_resource(self, filename: str):
        with resources.files(self.default_model_path).joinpath(filename).open('rb') as f:
            return pickle.load(f)

    def clean_and_scale_df(self):
        # Загружаем трансформер
        data_transformer = self._load_resource('data_transformer.pkl')
        # Обработка выбросов
        if self.clean_outliers:
            clf = self._load_resource('clf.pkl')
            outliers = clf.predict(self.chess_games_info)
        else:
            outliers = np.zeros(len(self.chess_games_info))
        df_no_out = self.chess_games_info[outliers == 0]
        self.outcome_column = df_no_out['outcome']
        self.cgi_no_out_norm = pd.DataFrame(data_transformer.fit_transform(df_no_out))

    def rating_predict(self):
        cat_boost_regressor = self._load_resource('cat_boost_regressor.pkl')
        rating_pred = cat_boost_regressor.predict(self.cgi_no_out_norm)
        return rating_pred

    def master_predict(self):
        knn_classificator = self._load_resource('knn_classificator.pkl')
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