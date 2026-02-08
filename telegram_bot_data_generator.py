from typing import Literal
import matplotlib
matplotlib.use('Agg')
from chess_data_fetch import ChesscomData
from postgresql_interaction import PopulateDB
from graph_visualization import *
from models import ChessModel
import pandas as pd


class TGBotDataGenerator:
    def __init__(self, language:str, nickname:str, messages:dict, room:str, blitz_num:int,
                 rapid_num:int, game_type:str, rating:int, calculate=True):
        self.language = language
        self.nickname = nickname
        self.messages = messages
        self.room = room
        self.blitz_num = blitz_num
        self.rapid_num = rapid_num
        self.game_type = game_type
        self.main_rating = rating
        self.calculate = calculate
        self.knight_bishop_coeff = 1.414423530036545643390348864151  # 1.5186327354661347360987111933486 - 1.3102143246069565506819865349535

        self.df_users = pd.DataFrame({'username': [self.nickname], 'num_games': [self.blitz_num+self.rapid_num]})
        self.chess_games_info = pd.DataFrame()
        self.games_by_moves = pd.DataFrame()
        self.achicode = []

        self.player_db_name = f'chess_{self.game_type}_{self.nickname.lower()}'
        self.all_db_name = f'chess_{self.game_type}'
        self.player_sql_db = PopulateDB(self.player_db_name)

        if self.calculate:
            self.generate_chess_data()
            self.run_scripts()

    def generate_chess_data(self):
        blitz_data = ChesscomData(username=self.nickname, num_games=self.blitz_num, game_type='blitz')
        rapid_data = ChesscomData(username=self.nickname, num_games=self.rapid_num, game_type='rapid')
        if self.blitz_num and self.rapid_num:
            self.chess_games_info = pd.concat([blitz_data.chesscom_df, rapid_data.chesscom_df], axis=0)
            self.games_by_moves = pd.concat([blitz_data.moves_df, rapid_data.moves_df], axis=0)
        elif self.blitz_num and not self.rapid_num:
            self.chess_games_info = blitz_data.chesscom_df
            self.games_by_moves = blitz_data.moves_df
        elif self.rapid_num and not self.blitz_num:
            self.chess_games_info = rapid_data.chesscom_df
            self.games_by_moves = rapid_data.moves_df
        else:
            raise ValueError('The number of games for either type (blitz or rapid) is not specified.')

    def run_scripts(self):
        self.player_sql_db.create_database()
        self.player_sql_db.save_df(df_users=self.df_users, games_info=self.chess_games_info,
                              games_by_moves=self.games_by_moves)
        self.player_sql_db.run_sql_script(script_path=fr'.\models\PostgreSQL_scripts\update_replace_neg1_with_null.sql')
        self.player_sql_db.run_sql_script(script_path=fr'.\models\PostgreSQL_scripts\add_new_columns.sql')
        self.player_sql_db.run_sql_script(script_path=fr'.\models\PostgreSQL_scripts\av_value_gen.sql')
        self.player_sql_db.run_sql_script(script_path=fr'.\models\PostgreSQL_scripts\update_replace_null_with_neg1.sql')

    def get_pieces_versus_sample(self):
        pieces_versus_sample_query = ('SELECT av_opening_knight_activ_coeff, av_mittelspiel_endgame_knight_activ_coeff, '
                                      ' av_opening_bishop_activ_coeff, av_mittelspiel_endgame_bishop_activ_coeff, '
                                      ' av_opening_mobility_inc, av_mittelspiel_endgame_mobility_inc, av_opening_mobility_dec, '
                                      ' av_mittelspiel_endgame_mobility_dec '
                                      'FROM games_info')
        pieces_versus_sample = self.player_sql_db.get_dataframe(pieces_versus_sample_query)
        pieces_columns = pd.DataFrame(columns=[
            'av_opening_knight_activ_coeff',
            'av_mittelspiel_endgame_knight_activ_coeff',
            'av_opening_bishop_activ_coeff',
            'av_mittelspiel_endgame_bishop_activ_coeff',
        ])
        activity_columns = pd.DataFrame(columns=[
            'av_opening_mobility_inc',
            'av_mittelspiel_endgame_mobility_inc',
            'av_opening_mobility_dec',
            'av_mittelspiel_endgame_mobility_dec'
        ])
        pieces_global_min = pieces_versus_sample[pieces_columns].min().min()
        pieces_global_max = pieces_versus_sample[pieces_columns].max().max()
        activity_global_min = pieces_versus_sample[activity_columns].min().min()
        activity_global_max = pieces_versus_sample[activity_columns].max().max()

        norm_func = lambda x: ((x - activity_global_min) * (pieces_global_max - pieces_global_min) / (activity_global_max - activity_global_min)) + activity_global_min
        pieces_versus_sample[activity_columns] = pieces_versus_sample[activity_columns].map(norm_func)
        return pieces_versus_sample

    def get_pieces_param_sample(self, game_phase: Literal['opening', 'mittelspiel_endgame']):
        av_player_query = (f'SELECT {self.knight_bishop_coeff} * AVG(av_{game_phase}_knight_activ_coeff) AS av_player_N, '
                                     f'AVG(av_{game_phase}_bishop_activ_coeff) AS av_player_B, '
                                     f'AVG(av_{game_phase}_rook_queen_activ_coeff) AS av_player_R_Q '
                                     f'FROM games_info')
        av_player_sample = self.player_sql_db.get_dataframe(av_player_query)
        av_player_dict = av_player_sample.to_dict('records')[0]

        pieces_param_sample_query = (f'SELECT {self.knight_bishop_coeff}*av_{game_phase}_knight_activ_coeff,'
                                     f'av_{game_phase}_bishop_activ_coeff, av_{game_phase}_rook_queen_activ_coeff '
                                     f'FROM games_info '
                                     f'WHERE main_rating >= {self.main_rating}-50 AND main_rating < {self.main_rating}+50 ')
        pieces_param_sample = PopulateDB(self.all_db_name).get_dataframe(pieces_param_sample_query)
        pieces_param_sample.columns = ['bishop activity', 'knight activity', 'rook & queen activity']
        return [pieces_param_sample, av_player_dict]

    def get_squares(self, is_captured=False):
        if is_captured:
                squares_query = ('SELECT white_move_index, black_move_index FROM games_by_moves WHERE move_is_capture IS TRUE')
        else:
            squares_query = ('SELECT white_move_index, black_move_index FROM games_by_moves')
        squares_df = self.player_sql_db.get_dataframe(squares_query)
        squares = squares_df.stack().reset_index(drop=True).astype(int)
        return squares

    def get_first_moves(self, turn_index:bool, num_plys=6):
        num_moves = int((num_plys + 1) / 2)
        first_moves_query = (f'SELECT white_move, black_move FROM games_by_moves '
                             f'WHERE move_number <= {num_moves} AND main_color = {turn_index}')
        first_moves_df = self.player_sql_db.get_dataframe(first_moves_query)
        first_moves_array = np.array(first_moves_df).reshape(-1, num_plys + num_plys%2)
        return first_moves_array

    def get_achicode(self) -> list[int]:
        achicode = []
        
        score_query = ('SELECT outcome, count(*) AS num FROM games_info '
                       'GROUP BY outcome '
                       'ORDER BY num DESC ')
        score_df = self.player_sql_db.get_dataframe(score_query)
        first_code_dict = {0: 0, 1: 1, 0.5: 2}
        first_code = first_code_dict[score_df['outcome'][0]]
        achicode.append(first_code)

        pieces_param_sample, av_player_dict = self.get_pieces_param_sample('opening')
        sorted_player_dict = dict(sorted(av_player_dict.items(), key=lambda item: item[1]))
        best_piece = next(iter(sorted_player_dict.keys()))
        second_code_dict = {'av_player_n': 0, 'av_player_b': 1, 'av_player_r_q':2}
        second_code = second_code_dict[best_piece]
        achicode.append(second_code)

        means = pieces_param_sample.mean()
        sum_of_means = means.sum()
        sum_of_means_by_player = sum(av_player_dict.values())
        third_code = 0 if sum_of_means > sum_of_means_by_player else 1
        achicode.append(third_code)

        castling_query = (
            'WITH game_stats AS ( '
            '   SELECT '
            '       id_game, '
            '       BOOL_OR(CASE '
            '           WHEN main_color IS FALSE AND white_move = \'O-O\' THEN TRUE '
            '           WHEN main_color IS TRUE AND black_move = \'O-O\' THEN TRUE '
            '           ELSE FALSE '
            '       END) AS has_short_castle, '
            '       BOOL_OR(CASE '
            '           WHEN main_color IS FALSE AND white_move = \'O-O-O\' THEN TRUE '
            '           WHEN main_color IS TRUE AND black_move = \'O-O-O\' THEN TRUE '
            '           ELSE FALSE '
            '       END) AS has_long_castle '
            '   FROM games_by_moves '
            '   GROUP BY id_game '
            ') '
            'SELECT '
            '   COUNT(DISTINCT CASE WHEN has_short_castle THEN id_game END) AS short_castle_games, '
            '   COUNT(DISTINCT CASE WHEN has_long_castle THEN id_game END) AS long_castle_games, '
            '   COUNT(DISTINCT CASE WHEN NOT has_short_castle AND NOT has_long_castle THEN id_game END) AS no_castle_games '
            'FROM game_stats;'
        )
        castling_df = self.player_sql_db.get_dataframe(castling_query)
        castling = castling_df.idxmax(axis=1).iloc[0]
        fourth_code_dict = {'short_castle_games': 0, 'long_castle_games': 1, 'no_castle_games': 2}
        fourth_code = fourth_code_dict[castling]
        achicode.append(fourth_code)

        inc_dec_query = ('SELECT AVG(av_mittelspiel_endgame_mobility_inc) AS av_inc, '
                         'AVG(av_mittelspiel_endgame_mobility_dec) AS av_dec '
                         'FROM games_info')
        inc_dec_df = self.player_sql_db.get_dataframe(inc_dec_query)
        fifth_code = 0 if inc_dec_df['av_inc'][0] >= inc_dec_df['av_dec'][0] else 1
        achicode.append(fifth_code)

        king_safety_query = ('SELECT AVG(av_mittelspiel_endgame_king_safety) AS king_safety '
                             'FROM games_info')
        king_safety_df = self.player_sql_db.get_dataframe(king_safety_query)
        av_king_safety_by_player = king_safety_df['king_safety'][0]
        av_king_safety = 31.867107450270755
        sixth_code = 0 if av_king_safety_by_player < av_king_safety else 1
        achicode.append(sixth_code)

        endgame_query = (
            'SELECT CAST(COUNT(CASE WHEN is_there_endgame THEN 1 END) AS FLOAT) / COUNT(*) AS endgame_percent '
            'FROM games_info')
        endgame_df = self.player_sql_db.get_dataframe(endgame_query)
        endgame_percent = endgame_df['endgame_percent'][0]
        seventh_code = 0 if endgame_percent <= 0.5 else 1
        achicode.append(seventh_code)
        return achicode

    def get_chess_model(self, clean_outliers=False):
        model_query = ('SELECT main_color, "opening_ACP", '
                       '"mittelspiel_and_endgame_ACP", "opening_STDPL", '
                       '"mittelspiel_and_endgame_STDPL", "opening_ACP_by_cauchy", '
                       '"mittelspiel_and_endgame_ACP_by_cauchy", "opening_STDPL_by_cauchy", '
                       '"mittelspiel_and_endgame_STDPL_by_cauchy", av_opening_mobility_inc, '
                       'av_mittelspiel_endgame_mobility_inc, av_opening_mobility_dec, '
                       'av_mittelspiel_endgame_mobility_dec, av_opening_king_safety, '
                       'av_mittelspiel_endgame_king_safety, av_opening_king_openness, '
                       'av_mittelspiel_endgame_king_openness, av_opening_knight_activ_coeff, '
                       'av_mittelspiel_endgame_knight_activ_coeff, av_opening_bishop_activ_coeff, '
                       'av_mittelspiel_endgame_bishop_activ_coeff, av_opening_rook_queen_activ_coeff, '
                       'av_mittelspiel_endgame_rook_queen_activ_coeff, av_mittelspiel_control, '
                       'av_endgame_control, av_opening_control, outcome '
                       'FROM games_info')
        model_df = self.player_sql_db.get_dataframe(model_query)
        chess_model = ChessModel(chess_games_info=model_df, game_type=self.game_type, clean_outliers=clean_outliers)
        rating_dict = chess_model.get_rating_dict()
        style_dict = chess_model.get_game_style()
        return [rating_dict, style_dict]


    def create_visualization(self, clean_outliers=False):
        storage = ChessStorage()
        rating_dict, style_dict = self.get_chess_model(clean_outliers=clean_outliers)
        storage.upload_json(style_dict, f'{self.nickname}/style_dict.json')
        OpeningTree(input_array=self.get_first_moves(turn_index=chess.WHITE), threshold=0, layer=5, user_name=self.nickname, turn='white', storage=storage)
        OpeningTree(input_array=self.get_first_moves(turn_index=chess.BLACK), threshold=0, layer=5, user_name=self.nickname, turn='black', storage=storage)
        for is_captured in [False, True]:
            squares = self.get_squares(is_captured=is_captured)
            description = ['all', 'is_captured'][is_captured]
            HeatBoard(username=self.nickname, squares=squares, description=description, storage=storage)
        for game_phase in ['opening', 'mittelspiel_endgame']:
            pieces_param_sample, av_player_dict = self.get_pieces_param_sample(game_phase)
            MarkedRaincloud(pieces_param_sample=pieces_param_sample, av_player_N=av_player_dict['av_player_n'],
                            av_player_B=av_player_dict['av_player_b'], av_player_R_Q=av_player_dict['av_player_r_q'],
                            main_rating=self.main_rating, username=self.nickname, game_phase=game_phase,
                            storage=storage)
        VersusViolin(username=self.nickname, sample=self.get_pieces_versus_sample(), storage=storage)
        AchievementsReport(username=self.nickname,
                           win_rating=rating_dict[1.0], draw_rating=rating_dict[0.5], lose_rating=rating_dict[0.0],
                           achicode=self.get_achicode(), language=self.language, storage=storage)
        return style_dict








