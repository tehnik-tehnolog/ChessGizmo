import chess
import mureq as mq
import pandas as pd
import chess.pgn
import io
import berserk
from chess_analyzer import EvalInfo, ModBoard
import os
from dotenv import load_dotenv

load_dotenv()


class BaseData:
    def __init__(self, username, num_games, game_type='blitz'):
        self.user_name = username
        self.num_games = num_games
        self.game_type = game_type
        self.chesscom_df = pd.DataFrame(columns=['id_game', 'status', 'id_player', 'id_opponent',
                                                 'main_color', 'main_rating', 'enemy_rating', 'outcome',
                                                 'opening_ACP', 'mittelspiel_and_endgame_ACP', 'is_there_endgame',
                                                 'opening_STDPL', 'mittelspiel_and_endgame_STDPL',
                                                 'opening_ACP_by_cauchy', 'mittelspiel_and_endgame_ACP_by_cauchy',
                                                 'opening_STDPL_by_cauchy', 'mittelspiel_and_endgame_STDPL_by_cauchy'])
        self.moves_df = pd.DataFrame(columns=['id_game', 'move_number', 'main_color',
                                              'white_move', 'black_move',  'white_move_index',  'black_move_index',
                                              'move_is_capture', 'analysis',
                                              'CP_loss', 'CP_loss_by_cauchy', 'pieces_material', 'pawns','game_phase',
                                              'mobility_increment', 'mobility_decrement','control',
                                              'king_safety', 'king_openness', 'knight_activity_coeff',
                                              'bishop_activity_coeff', 'rook_and_queen_activity_coeff'])


class PlayerInfo:
    def __init__(self, output_username):
        self.output_username = output_username
        self.username = None
        self.blitz_num = 0
        self.rapid_num = 0
        self.blitz_rating = 0
        self.rapid_rating = 0

        self.get_username()
        self.get_stats()

    def get_username(self):
        player_request = f'https://api.chess.com/pub/player/{self.output_username.lower()}'
        self.username = mq.get(player_request).json()['url'][29:]

    def get_stats(self):
        stats_request = f'https://api.chess.com/pub/player/{self.output_username.lower()}/stats'
        stats_dict = mq.get(stats_request).json()
        if 'chess_blitz' in stats_dict.keys():
            self.blitz_num = sum(stats_dict['chess_blitz']['record'].values())
            self.blitz_rating = stats_dict['chess_blitz']['last']['rating']

        if 'chess_rapid' in stats_dict.keys():
            self.rapid_num = sum(stats_dict['chess_rapid']['record'].values())
            self.rapid_rating = stats_dict['chess_rapid']['last']['rating']


class ChesscomData(BaseData):
    def __init__(self, username, num_games, game_type='blitz'):
        super().__init__(username=username, num_games=num_games)
        self.username = username
        self.num_games = num_games
        self.moves_df.loc[0] = pd.Series()  # add first empty row
        self.add_new_row = True

        self.archives_url = f'https://api.chess.com/pub/player/{self.username.lower()}/games/archives'
        self.archives = iter(reversed(mq.get(self.archives_url).json()["archives"]))

        self.color = None
        self.main_color_index = None
        self.status = None
        self.main_rating = None
        self.enemy_rating = None
        self.outcome = 0.5
        self.opponent = None
        self.enemy_move = None
        self.game_type = game_type

        self.generate_data()

    opposite = lambda self, color: 'white' if color == 'black' else 'black'

    color_index = lambda self, color: chess.WHITE if color == 'white' else chess.BLACK

    def get_side_info(self, game):
        if self.username.lower() == game['white']['username'].lower():
            self.color = 'white'
        else:
            self.color = 'black'

        self.main_color_index = self.color_index(self.color)

        if game[self.color]['result'] == 'win':
            self.outcome = 1.
            self.status = game['black']['result']
        elif game[self.opposite(self.color)]['result'] == 'win':
            self.outcome = 0.
            self.status = game['white']['result']
        else:
            self.outcome = 0.5
            self.status = game[self.color]['result']

        self.main_rating = game[self.color]['rating']
        self.enemy_rating = game[self.opposite(self.color)]['rating']
        self.username = game[self.color]['username']
        self.opponent = game[self.opposite(self.color)]['username']

    @staticmethod
    def delta_rating(game):
        return abs(game['white']['rating'] - game['black']['rating'])

    def _generate_moves_data(self, id_game, game, main_color):
        pgn_string = io.StringIO(game['pgn'])
        game_pgn = chess.pgn.read_game(pgn_string)
        color_index = self.color_index(main_color)
        board = ModBoard()
        self.add_new_row = True
        for ply, move in enumerate(list(game_pgn.mainline_moves())):
            side = board.board.turn
            is_capture = board.board.is_capture(move)
            san_move = board.board.san(move)
            board.push_move(move)
            m_len = self.moves_df.index.size - 1  # length

            if ~ ply & 1 == color_index:
                # m_len = self.moves_df.index.size - 1  # length
                self.moves_df.at[m_len, 'id_game'] = id_game
                self.moves_df.at[m_len, 'move_number'] = int((ply + 2) / 2)
                self.moves_df.at[m_len, f'{main_color}_move'] = san_move
                self.moves_df.at[m_len, f'{main_color}_move_index'] = move.to_square
                self.moves_df.at[m_len, 'move_is_capture'] = is_capture
                # ------------------------------------------------------------------------------------------------
                board.comp_CP_analysis()
                self.moves_df.at[m_len, 'analysis'] = board.analysis
                self.moves_df.at[m_len, 'CP_loss'] = board.delta_CP
                self.moves_df.at[m_len, 'CP_loss_by_cauchy'] = board.delta_CP_by_cauchy
                material_info = board.material(side)
                self.moves_df.at[m_len, 'pieces_material'] = material_info['pieces_material']
                self.moves_df.at[m_len, 'pawns'] = material_info['pawns']
                self.moves_df.at[m_len, 'game_phase'] = board.game_phase(ply)
                board.mobility()
                self.moves_df.at[m_len, 'mobility_increment'] = board.mobility_increment
                self.moves_df.at[m_len, 'mobility_decrement'] = board.mobility_decrement
                self.moves_df.at[m_len, 'control'] = board.control()
                self.moves_df.at[m_len, 'king_safety'] = board.king_control_zone_safety(side)
                self.moves_df.at[m_len, 'king_openness'] = board.king_openness(side)
                pu_dict = board.pieces_usefulness_dict()
                self.moves_df.at[m_len, 'knight_activity_coeff'] = pu_dict['N']
                self.moves_df.at[m_len, 'bishop_activity_coeff'] = pu_dict['B']
                self.moves_df.at[m_len, 'rook_and_queen_activity_coeff'] = pu_dict['R_Q']

                if self.add_new_row is False:
                    self.moves_df.loc[m_len + 1] = pd.Series()
                    self.add_new_row = True
                elif self.add_new_row is True:
                    self.add_new_row = False

            else:

                self.enemy_move = san_move
                enemy_color = self.opposite(main_color)
                self.moves_df.at[m_len, f'{enemy_color}_move'] = self.enemy_move
                self.moves_df.at[m_len, f'{enemy_color}_move_index'] = move.to_square
                board.mobility()
                board.comp_CP_analysis()

                if self.add_new_row is False:
                    self.moves_df.loc[m_len + 1] = pd.Series()
                    self.add_new_row = True
                elif self.add_new_row is True:
                    self.add_new_row = False

    def _generate_data(self, games):
        while ((game := next(games, None)) is not None) and not self.chesscom_df.index.size == self.num_games:
            if game['rated'] and game['time_class'] == self.game_type and game['rules'] == 'chess' and \
                    self.delta_rating(game) <= 450 and game['pgn'].__contains__('15. '):
                self.chesscom_df.loc[len(self.chesscom_df.index)] = pd.Series()
                id_game = game['url'][32:]
                g_len = self.chesscom_df.index.size - 1
                self.chesscom_df.at[g_len, 'id_game'] = id_game

                self.get_side_info(game)
                self.chesscom_df.at[g_len, 'status'] = self.status
                self.chesscom_df.at[g_len, 'id_player'] = self.username
                self.chesscom_df.at[g_len, 'id_opponent'] = self.opponent
                self.chesscom_df.at[g_len, 'main_color'] = self.main_color_index
                self.chesscom_df.at[g_len, 'main_rating'] = self.main_rating
                self.chesscom_df.at[g_len, 'enemy_rating'] = self.enemy_rating
                self.chesscom_df.at[g_len, 'outcome'] = self.outcome

                self._generate_moves_data(id_game, game, self.color)
                self.moves_df.loc[self.moves_df['id_game'] == id_game, 'main_color'] = self.main_color_index
                self.moves_df.drop(self.moves_df.tail(1).index, inplace=True)  # drop last row

                eval_info = EvalInfo(self.moves_df, 'CP_loss')
                self.chesscom_df.at[g_len, 'opening_ACP'] = eval_info.opening_acp
                self.chesscom_df.at[g_len, 'mittelspiel_and_endgame_ACP'] = eval_info.mittel_end_spiel_acp
                self.chesscom_df.at[g_len, 'is_there_endgame'] = eval_info.is_there_endgame
                self.chesscom_df.at[g_len, 'opening_STDPL'] = eval_info.opening_stdpl
                self.chesscom_df.at[g_len, 'mittelspiel_and_endgame_STDPL'] = eval_info.mittel_end_spiel_stdpl

                eval_info_by_cauchy = EvalInfo(self.moves_df, 'CP_loss_by_cauchy')
                self.chesscom_df.at[g_len, 'opening_ACP_by_cauchy'] = eval_info_by_cauchy.opening_acp
                self.chesscom_df.at[
                    g_len, 'mittelspiel_and_endgame_ACP_by_cauchy'] = eval_info_by_cauchy.mittel_end_spiel_acp
                self.chesscom_df.at[g_len, 'is_there_endgame'] = eval_info_by_cauchy.is_there_endgame
                self.chesscom_df.at[g_len, 'opening_STDPL_by_cauchy'] = eval_info_by_cauchy.opening_stdpl
                self.chesscom_df.at[
                    g_len, 'mittelspiel_and_endgame_STDPL_by_cauchy'] = eval_info_by_cauchy.mittel_end_spiel_stdpl

                print(f'{self.username} is done. ID: {id_game}. Num. {self.chesscom_df.index.size}')

    def generate_data(self):
        while ((archive := next(self.archives, None)) is not None) and \
                not self.chesscom_df.index.size == self.num_games:
            games = iter(mq.get(archive).json()["games"])
            self._generate_data(games)
        if not self.chesscom_df.index.size == self.num_games:
            return f'not enough data by {self.username}'


class LichessData(BaseData):
    def __init__(self, username, num_games):
        super().__init__(username=username, num_games=num_games)

        self.username = username
        self.num_games = num_games

        self.token = os.getenv('LICHESS_TOKEN')
        self.games = None

        self.main_color_lst = []
        self.main_rating_lst = []
        self.enemy_rating_lst = []
        self.opponent_lst = []

        self.get_games_pgn()
        self.get_games_info()

    def get_games_pgn(self):
        with berserk.TokenSession(self.token) as session:
            client = berserk.Client(session=session)
        self.games = client.games.export_by_player(username=self.username, as_pgn=True, rated=True, max=self.num_games,
                                                   perf_type='rapid',
                                                   clocks=False,
                                                   opening=True,
                                                   evals=False, ongoing=False, finished=True)

    def get_games_info(self):
        while ((game := next(self.games, None)) is not None):
            self.chesscom_df.loc[len(self.chesscom_df.index)] = pd.Series()
            g_len = self.chesscom_df.index.size - 1
            pgn_string = io.StringIO(game)
            game_pgn = chess.pgn.read_game(pgn_string)
            game_info = dict(game_pgn.headers)
            self.chesscom_df.at[g_len, 'id_game'] = game_info['Site'][20:]
            main_color_is_white = game_info['White'].lower() == self.username.lower()
            white_win = game_info['Result'] == '1-0'
            black_win = game_info['Result'] == '0-1'
            if (main_color_is_white and white_win) or (not main_color_is_white and black_win):
                self.chesscom_df.at[g_len, 'status'] = 'Win'
                self.chesscom_df.at[g_len, 'outcome'] = 1
            elif (main_color_is_white and black_win) or (not main_color_is_white and white_win):
                self.chesscom_df.at[g_len, 'status'] = 'Lose'
                self.chesscom_df.at[g_len, 'outcome'] = 0
            else:
                self.chesscom_df.at[g_len, 'status'] = 'Draw'
                self.chesscom_df.at[g_len, 'outcome'] = 0.5
            self.chesscom_df.at[g_len, 'id_player'] = self.username
            if main_color_is_white:
                self.chesscom_df.at[g_len, 'main_color'] = 1  # chess.White
                self.chesscom_df.at[g_len, 'id_opponent'] = game_info['Black']
                self.chesscom_df.at[g_len, 'main_rating'] = game_info['WhiteElo']
                self.chesscom_df.at[g_len, 'enemy_rating'] = game_info['BlackElo']
            else:
                self.chesscom_df.at[g_len, 'main_color'] = 0  # chess.Black
                self.chesscom_df.at[g_len, 'id_opponent'] = game_info['White']
                self.chesscom_df.at[g_len, 'main_rating'] = game_info['BlackElo']
                self.chesscom_df.at[g_len, 'enemy_rating'] = game_info['WhiteElo']


class FromDataBase(BaseData):
    def __init__(self, input_df: pd.DataFrame, username=None, num_games=None):
        super().__init__(username=username, num_games=num_games)
        self.input_df = input_df
        if username is not None:
            self.input_df = self.input_df[self.input_df['player'] == username]
        if num_games is not None:
            self.input_df = self.input_df.head(num_games)
        self.num_games = num_games
        self.populate_df()

    def populate_df(self):
        self.chesscom_df['id_game'] = self.input_df.index
        self.chesscom_df['status'] = self.input_df['result']
        self.chesscom_df['id_player'] = self.input_df['player']
        self.chesscom_df['id_opponent'] = self.input_df['opponent']
        self.chesscom_df['main_color'] = self.input_df['color'].apply(lambda x: x.lower())
        self.chesscom_df['main_rating'] = self.input_df['player_Elo']
        self.chesscom_df['enemy_rating'] = self.input_df['opponent_Elo']
        outcome_dict = {'Win': 1, 'Lose': 0, 'Draw': 0.5}
        self.chesscom_df['outcome'] = self.chesscom_df['status'].apply(lambda x: outcome_dict[x])
        self.games = iter(self.input_df['lines'].apply(lambda pgn: self.get_chess_pgn(str(pgn))).to_list())

    def get_chess_pgn(self, pgn: str):  # => chess.pgn
        pgn_string = io.StringIO(pgn)
        game = chess.pgn.read_game(pgn_string)
        return game