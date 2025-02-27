from typing import Union, Literal

# import cairocffi as cairo
import cairosvg
import chess
import matplotlib
matplotlib.use('Agg')
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from io import BytesIO

from chess_data_fetch import PlayerInfo
from chess_data_fetch import ChesscomData
from mysql_interaction import check_database_exists, PopulateDB
from graph_visualization import *
from config import HOST, USER, PASSWORD
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
        self.player_sql_db.run_sql_script(script_path=fr'.\models\MySQL_scriprts\update_replace_neg1_with_null.sql')
        self.player_sql_db.run_sql_script(script_path=fr'.\models\MySQL_scriprts\add_new_columns.sql')
        self.player_sql_db.run_sql_script(script_path=fr'.\models\MySQL_scriprts\av_value_gen.sql')
        # player_sql_db.run_sql_script(script_path=fr'.\models\MySQL_scriprts\update_replace_null_with_neg1.sql')

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
            squares_query = (f'SELECT white_move_index, black_move_index FROM games_by_moves WHERE is_captured = 1')
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

    def create_visualization(self):
        OpeningTree(input_array=self.get_first_moves(turn_index=chess.WHITE), threshold=0, layer=5, user_name=nickname, turn='white')
        OpeningTree(input_array=self.get_first_moves(turn_index=chess.BLACK), threshold=0, layer=5, user_name=nickname, turn='black')
        for is_captured in [False, False]:
            squares = self.get_squares(is_captured=is_captured)
            HeatBoard(username=self.nickname, squares=squares)
        for game_phase in ['opening', 'mittelspiel_endgame']:
            pieces_param_sample, av_player_dict = self.get_pieces_param_sample(game_phase)
            MarkedRaincloud(pieces_param_sample=pieces_param_sample, av_player_N=av_player_dict['av_player_N'],
                            av_player_B=av_player_dict['av_player_B'], av_player_R_Q=av_player_dict['av_player_R_Q'],
                            main_rating=self.main_rating, username=self.nickname, game_phase=game_phase)
        VersusViolin(username=self.nickname, sample=self.get_pieces_versus_sample())
        #AchievementsReport(username=self.nickname, win_rating=, draw_rating=, lose_rating=, achicode=[], language=self.language)








