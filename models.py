#from pyod.models.ecod import ECOD
#from sklearn.preprocessing import PowerTransformer, OrdinalEncoder
#from sklearn.pipeline import Pipeline
import pickle
import pandas as pd
from mysql_interaction import PopulateDB

df = PopulateDB('chess_rapid').get_dataframe(f"Select * from chess_games_info where id_player = 'JIOBU_KOH9I'")
df = df.loc[:, ['main_color', 'opening_ACP',
                'mittelspiel_and_endgame_ACP', 'opening_STDPL',
                'mittelspiel_and_endgame_STDPL', 'opening_ACP_by_cauchy',
                'mittelspiel_and_endgame_ACP_by_cauchy', 'opening_STDPL_by_cauchy',
                'mittelspiel_and_endgame_STDPL_by_cauchy', 'av_opening_mobility_inc',
                'av_mittelspiel_endgame_mobility_inc', 'av_opening_mobility_dec',
                'av_mittelspiel_endgame_mobility_dec', 'av_opening_king_safety',
                'av_mittelspiel_endgame_king_safety', 'av_opening_king_openness',
                'av_mittelspiel_endgame_king_openness', 'av_opening_knight_activ_coeff',
                'av_mittelspiel_endgame_knight_activ_coeff', 'av_opening_bishop_activ_coeff',
                'av_mittelspiel_endgame_bishop_activ_coeff', 'av_opening_rook_queen_activ_coeff',
                'av_mittelspiel_endgame_rook_queen_activ_coeff', 'av_mittelspiel_control',
                'av_endgame_control', 'av_opening_control', 'outcome']]


game_type = 'blitz'
model_path = f'./models/{game_type}'


def clean_and_scale_df(df:pd.DataFrame, model_path:str,) -> [pd.DataFrame, pd.Series]:
    with open(model_path + '/clf.pkl', 'rb') as clf_file:
        clf = pickle.load(clf_file)

    with open(model_path+'/data_transformer.pkl', 'rb') as data_transformer_file:
        data_transformer = pickle.load(data_transformer_file)


    outliers = clf.predict(df)
    df_no_out = df[outliers == 0]
    outcome_column = df_no_out['outcome']
    df_no_out_norm = pd.DataFrame(data_transformer.fit_transform(df_no_out))
    return [df_no_out_norm, outcome_column]


cgi_no_out_norm, outcome_column = clean_and_scale_df(df=df, model_path=model_path)

with open(model_path + f'/knn_classificator.pkl', 'rb') as knn_file:
    knn_classificator = pickle.load(knn_file)

with open(model_path + f'/catboost_regression.pkl', 'rb') as cbr_file:
    cat_boost_regressor = pickle.load(cbr_file)

rating_pred = cat_boost_regressor.predict(cgi_no_out_norm)
master_pred = knn_classificator.predict(cgi_no_out_norm.iloc[:, 1:9])
cgi_no_out_norm.columns = df.columns
cgi_no_out_norm['outcome'] = outcome_column
cgi_no_out_norm['rating_pred'] = rating_pred
cgi_no_out_norm['master_pred'] = master_pred

outcome_dict = cgi_no_out_norm.loc[:, ['outcome', 'rating_pred']].groupby('outcome').mean().to_dict()
for key in [1.0, 0.0, 0.5]:
    outcome_dict.setdefault(key, None)

print(cgi_no_out_norm.loc[:, ['master_pred', 'outcome']].groupby('master_pred').count())
print(cgi_no_out_norm['master_pred'].count())
outcome_dict = {1.: None, .5: None, 0.: None}