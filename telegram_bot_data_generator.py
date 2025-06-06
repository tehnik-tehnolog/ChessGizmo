from typing import Union, Literal, Any
import matplotlib
from scipy.stats import describe

matplotlib.use('Agg')

from chess_data_fetch import PlayerInfo
from chess_data_fetch import ChesscomData
from mysql_interaction import check_database_exists, PopulateDB
from graph_visualization import *
from config import HOST, USER, PASSWORD, PROJECT_PATH
import pandas as pd


language = 'RU_ru'
messages = 'messages'
room = 'Chesscom'
nickname = 'wsenorm'
blitz_num = 5
rapid_num = 0
game_type = 'blitz'
blitz_rating = 500
rapid_rating = 1000

main_rating = blitz_rating if game_type == 'blitz' else rapid_rating




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

        self.chess_df_users = pd.DataFrame({'username': [self.nickname], 'num_games': [self.blitz_num+self.rapid_num]})
        self.chess_games_info = pd.DataFrame()
        self.games_by_moves = pd.DataFrame()
        self.achicode = []

        self.player_db_name = f'chess_{self.game_type}_{self.nickname}'
        self.all_db_name = f'chess_{self.game_type}'
        self.player_sql_db = PopulateDB(self.player_db_name)

        if self.calculate:
            self.generate_chess_data()
            self.run_scripts()

    def generate_chess_data(self):
        blitz_data = ChesscomData(username=self.nickname, num_games=self.blitz_num, game_type='blitz')
        rapid_data = ChesscomData(username=self.nickname, num_games=self.rapid_num, game_type='rapid')
        self.chess_games_info = pd.concat([blitz_data.chesscom_df, rapid_data.chesscom_df], axis=0)
        self.games_by_moves = pd.concat([blitz_data.moves_df, rapid_data.moves_df], axis=0)

    def run_scripts(self):
        self.player_sql_db.create_database()
        self.player_sql_db.save_df(chess_df_users=self.chess_df_users, chess_games_info=self.chess_games_info,
                              games_by_moves=self.games_by_moves)
        self.player_sql_db.run_sql_script(script_path=PROJECT_PATH+fr'\models\MySQL_scriprts\update_replace_neg1_with_null.sql')
        self.player_sql_db.run_sql_script(script_path=PROJECT_PATH+fr'\models\MySQL_scriprts\add_new_columns.sql')
        self.player_sql_db.run_sql_script(script_path=PROJECT_PATH+fr'\models\MySQL_scriprts\av_value_gen.sql')
        #self.player_sql_db.run_sql_script(script_path=PROJECT_PATH+fr'\models\MySQL_scriprts\update_replace_null_with_neg1.sql')

    def get_pieces_versus_sample(self):
        pieces_versus_sample_query = ('SELECT av_opening_knight_activ_coeff, av_mittelspiel_endgame_knight_activ_coeff, '
                                      ' av_opening_bishop_activ_coeff, av_mittelspiel_endgame_bishop_activ_coeff, '
                                      ' av_opening_mobility_inc, av_mittelspiel_endgame_mobility_inc, av_opening_mobility_dec, '
                                      ' av_mittelspiel_endgame_mobility_dec '
                                      'FROM chess_games_info')
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
        pieces_versus_sample[activity_columns] = pieces_versus_sample[activity_columns].applymap(norm_func)
        return pieces_versus_sample

    def get_pieces_param_sample(self, game_phase: Literal['opening', 'mittelspiel_endgame']):
        av_player_query = (f'SELECT {self.knight_bishop_coeff} * AVG(av_{game_phase}_knight_activ_coeff) AS av_player_N, '
                                     f'AVG(av_{game_phase}_bishop_activ_coeff) AS av_player_B, '
                                     f'AVG(av_{game_phase}_rook_queen_activ_coeff) AS av_player_R_Q '
                                     f'FROM chess_games_info')
        av_player_sample = self.player_sql_db.get_dataframe(av_player_query)
        av_player_dict = av_player_sample.to_dict('records')[0]

        pieces_param_sample_query = (f'SELECT {self.knight_bishop_coeff}*av_{game_phase}_knight_activ_coeff,'
                                     f'av_{game_phase}_bishop_activ_coeff, av_{game_phase}_rook_queen_activ_coeff '
                                     f'FROM chess_games_info '
                                     f'WHERE main_rating >= {self.main_rating}-50 AND main_rating < {self.main_rating}+50 ')
        pieces_param_sample = PopulateDB(self.all_db_name).get_dataframe(pieces_param_sample_query)
        pieces_param_sample.columns = ['bishop activity', 'knight activity', 'rook & queen activity']
        return [pieces_param_sample, av_player_dict]

    def get_squares(self, is_captured=False):
        if is_captured:
            squares_query = (f'SELECT white_move_index, black_move_index FROM games_by_moves WHERE move_is_capture = 1')
        else:
            squares_query = (f'SELECT white_move_index, black_move_index FROM games_by_moves')
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
        
        score_query = ('SELECT outcome, count(*) AS num FROM chess_games_info '
                       'GROUP BY outcome '
                       'ORDER BY num DESC ')
        score_df = self.player_sql_db.get_dataframe(score_query)
        first_code_dict = {0: 0, 1: 1, 0.5: 2}
        first_code = first_code_dict[score_df['outcome'][0]]
        achicode.append(first_code)

        pieces_param_sample, av_player_dict = self.get_pieces_param_sample('opening')
        sorted_player_dict = dict(sorted(av_player_dict.items(), key=lambda item: item[1]))
        best_piece = next(iter(sorted_player_dict.keys()))
        second_code_dict = {'av_player_N': 0, 'av_player_B': 1, 'av_player_R_Q':2}
        second_code = second_code_dict[best_piece]
        achicode.append(second_code)

        means = pieces_param_sample.mean()
        sum_of_means = means.sum()
        sum_of_means_by_player = sum(av_player_dict.values())
        third_code = 0 if sum_of_means > sum_of_means_by_player else 1
        achicode.append(third_code)

        castling_query = (
            'SELECT '
            '   COUNT(DISTINCT IF(has_short_castle, id_game, NULL)) AS short_castle_games, '
            '   COUNT(DISTINCT IF(has_long_castle, id_game, NULL)) AS long_castle_games, '
            '   COUNT(DISTINCT IF(NOT has_short_castle AND NOT has_long_castle, id_game, NULL)) AS no_castle_games '
            'FROM ( '
            '   SELECT '
            '       id_game, '
            '       MAX( '
            '           CASE main_color '
            '               WHEN 0 THEN white_move = \'O-O\' '
            '               WHEN 1 THEN black_move = \'O-O\' '
            '           END '
            '       ) AS has_short_castle, '
            '       MAX( '
            '           CASE main_color '
            '               WHEN 0 THEN white_move = \'O-O-O\' '
            '               WHEN 1 THEN black_move = \'O-O-O\' '
            '           END '
            '       ) AS has_long_castle '
            '   FROM games_by_moves '
            '   GROUP BY id_game '
            ') AS game_stats;'
        )
        castling_df = self.player_sql_db.get_dataframe(castling_query)
        castling = castling_df.idxmax(axis=1).iloc[0]
        fourth_code_dict = {'short_castle_games': 0, 'long_castle_games': 1, 'no_castle_games': 2}
        fourth_code = fourth_code_dict[castling]
        achicode.append(fourth_code)

        inc_dec_query = ('SELECT AVG(av_mittelspiel_endgame_mobility_inc) AS av_inc, '
                         'AVG(av_mittelspiel_endgame_mobility_dec) AS av_dec '
                         'FROM chess_games_info')
        inc_dec_df = self.player_sql_db.get_dataframe(inc_dec_query)
        fifth_code = 0 if inc_dec_df['av_inc'][0] >= inc_dec_df['av_dec'][0] else 1
        achicode.append(fifth_code)

        king_safety_query = ('SELECT AVG(av_mittelspiel_endgame_king_safety) AS king_safety '
                             'FROM chess_games_info')
        king_safety_df = self.player_sql_db.get_dataframe(king_safety_query)
        av_king_safety_by_player = king_safety_df['king_safety'][0]
        av_king_safety = 31.867107450270755
        sixth_code = 0 if av_king_safety_by_player < av_king_safety else 1
        achicode.append(sixth_code)

        endgame_query = ('SELECT AVG(is_there_endgame) AS endgame_percent '
                         'FROM chess_games_info')
        endgame_df = self.player_sql_db.get_dataframe(endgame_query)
        endgame_percent = endgame_df['endgame_percent'][0]
        seventh_code = 0 if endgame_percent <= 0.5 else 1
        achicode.append(seventh_code)
        return achicode

    def create_visualization(self):
        OpeningTree(input_array=self.get_first_moves(turn_index=chess.WHITE), threshold=0, layer=5, user_name=nickname, turn='white')
        OpeningTree(input_array=self.get_first_moves(turn_index=chess.BLACK), threshold=0, layer=5, user_name=nickname, turn='black')
        for is_captured in [False, True]:
            squares = self.get_squares(is_captured=is_captured)
            description = ['all', 'is_captured'][is_captured]
            HeatBoard(username=self.nickname, squares=squares, description=description)
        for game_phase in ['opening', 'mittelspiel_endgame']:
            pieces_param_sample, av_player_dict = self.get_pieces_param_sample(game_phase)
            MarkedRaincloud(pieces_param_sample=pieces_param_sample, av_player_N=av_player_dict['av_player_N'],
                            av_player_B=av_player_dict['av_player_B'], av_player_R_Q=av_player_dict['av_player_R_Q'],
                            main_rating=self.main_rating, username=self.nickname, game_phase=game_phase)
        VersusViolin(username=self.nickname, sample=self.get_pieces_versus_sample())
        AchievementsReport(username=self.nickname, win_rating=800, draw_rating=1230, lose_rating=1896, achicode=self.get_achicode(), language=self.language)








