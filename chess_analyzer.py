import pandas as pd
import chess
import chess.pgn
from stockfish import Stockfish


class EvalInfo:
    def __init__(self, CP_loss_df: pd.DataFrame, column: str):
        self.CP_loss_df = CP_loss_df
        self.column = column

        self.opening_moves = None
        self.mittel_end_spiel_moves = None
        self.is_there_endgame = False

        self.opening_acp = 0
        self.mittel_end_spiel_acp = 0

        self.opening_stdpl = 0
        self.mittel_end_spiel_stdpl = 0

        self.generate_ACP_info()
        self.generate_STDPL_info()

    @property
    def acp(self):
        return self.comp_ACP(self.CP_loss_df[f'{self.column}'].tolist())

    @property
    def stdcpl(self):
        return self.comp_STDCPL(self.CP_loss_df[f'{self.column}'].tolist(), self.acp())

    def comp_ACP(self, part_of_game):
        return sum(part_of_game) / len(part_of_game)

    def generate_ACP_info(self):
        opening_fil = self.CP_loss_df["game_phase"] == "opening"
        mittelspiel_fil = self.CP_loss_df["game_phase"] != "opening"
        endgame_fil = self.CP_loss_df["game_phase"] == "endgame"

        self.opening_moves = self.CP_loss_df[f'{self.column}'].loc[opening_fil].to_list()
        self.mittel_end_spiel_moves = self.CP_loss_df[f'{self.column}'].loc[mittelspiel_fil].to_list()

        self.opening_acp = self.comp_ACP(self.opening_moves)

        if not len(self.mittel_end_spiel_moves):
            self.mittel_end_spiel_acp = 0
        else:
            self.mittel_end_spiel_acp = self.comp_ACP(self.mittel_end_spiel_moves)

        self.is_there_endgame = endgame_fil.any()

    def comp_STDCPL(self, part_of_game, acp):
        part_len = len(part_of_game)
        return sum(list(map(lambda x: ((x - acp) ** 2) / part_len ** .5, part_of_game)))

    def generate_STDPL_info(self):
        self.opening_stdpl = self.comp_STDCPL(self.opening_moves, self.opening_acp)

        if self.mittel_end_spiel_acp == 0:
            self.mittel_end_spiel_stdpl = 0
        else:
            self.mittel_end_spiel_stdpl = self.comp_STDCPL(self.mittel_end_spiel_moves, self.mittel_end_spiel_acp)


class ModBoard:
    def __init__(self):
        # super().__init__(self.moves, self.game_link, self.analyze_a_game)
        # __eval__ = EvalInfo(game_link, analyze_a_game=False)
        # self.moves = __eval__.moves
        self.board = chess.Board()
        # Piece Values in centipawns
        self.pawn_value = 100
        self.knight_value = 315
        self.bishop_value = 333
        self.rook_value = 563
        self.queen_value = 880
        self.king_value = 1  # maybe 400 or 220 or 2500

        self.value_of_attacks_compliance = {'N': 20, 'B': 20, 'R': 40, 'Q': 80}
        self.analysis = 27  # starting position evaluation
        self.delta_CP = 0  # change in eval per ply
        self.delta_CP_by_cauchy = 0

        self.stockfish = Stockfish(
            path=r'.\stockfish_with_avx2\stockfish\stockfish-windows-x86-64-avx2.exe')
        """Cauchy scale parameters."""
        self.gamma_high = 0.3183
        self.gamma_width = 100

        """Start material."""
        self.pawns_num = 8
        self.knights_num = 2
        self.bishops_num = 2
        self.rooks_num = 2
        self.queens_num = 1

        self.all_material = 4103  # material from the starting position by one side
        self.pieces_material = 3303  # material only pieces from the starting position by one side
        """ about endgame point material
            Only 52.24% of chess games reach the endgame.
            SOURCE: Fundamental Chess Endings by GM Muller & IM Lamprecht © 2001
            The book uses a dataset ChessBase Mega Database Collection 2001
            Where the rating range of 50-percentile players is 2400 - 2900. If we take a sample from my dataset
            with such a range, then with a material value of 965 centipawns,
            the number of games that reach the endgame will be 52%. I took 965 centipawns as the endgame point.

            1172(не учтитывая пешки) если брать среднеевзвешенное по типам окончаний и их вероятности
            """
        self.endgame_point_material = 710  # 965
        self.main_mobility = 20  # number of possible moves from the starting position
        self.enemy_mobility = 20  # similarly main_mobility

        self.mobility_increment = 0  # the extent to which my position has improved due to my move
        self.mobility_decrement = 0  # how much the enemy position improved due to my move

        # self.comp_CP_analysis()
        self.material(self.board.turn)
        # mobility maybe ??

    white_or_black = lambda self, color: chess.WHITE if color == 'white' else chess.BLACK

    sign = lambda self, x: (x / abs(x)) if x else 0

    BB_WHITE_KING_CONTROL_ZONE = [chess._step_attacks(sq, [17, 16, 15, 9, 8, 7, 1, 0, -9, -8, -7, -1]) for sq in
                                  chess.SQUARES]

    BB_BLACK_KING_CONTROL_ZONE = [chess._step_attacks(sq, [9, 8, 7, 1, 0, -9, -8, -7, -1, -17, -16, -15]) for sq in
                                  chess.SQUARES]

    piece_mask_map = lambda self: {'N': self.board.knights, 'B': self.board.bishops, 'R': self.board.rooks,
                                   'Q': self.board.queens, 'K': self.board.knights}

    def push_move(self, move):
        self.board.push(move)

    def material(self, side):
        # Material
        self.pawns_num = len(self.board.pieces(1, side))
        self.knights_num = len(self.board.pieces(2, side))
        self.bishops_num = len(self.board.pieces(3, side))
        self.rooks_num = len(self.board.pieces(4, side))
        self.queens_num = len(self.board.pieces(5, side))

        self.pieces_material = self.knight_value * self.knights_num + self.bishop_value * self.bishops_num + \
                               self.rook_value * self.rooks_num + self.queen_value * self.queens_num + self.king_value
        self.all_material = self.pieces_material + self.pawns_num * 100
        return {'is a queen': self.queens_num, 'pieces_material': self.pieces_material, 'pawns': self.pawns_num,
                'all_material': self.all_material}

    def weight_by_cauchy_distribution(self, cp: float):
        pi = 3.1415926535898
        return 1 / (pi * self.gamma_high * (1 + (cp / self.gamma_width) ** 2))

    def comp_CP_analysis(self):
        self.stockfish.set_fen_position(fen_position=self.board.fen())
        CP_eval = self.stockfish.get_evaluation()
        previous_analysis = self.analysis
        if CP_eval['type'] == 'cp':
            self.analysis = CP_eval['value']
        elif CP_eval['type'] == 'mate':
            self.analysis = self.sign(CP_eval['value']) * 1000  # like chess.com does
        self.delta_CP = self.analysis - previous_analysis
        self.delta_CP_by_cauchy = self.delta_CP * self.weight_by_cauchy_distribution(
            (self.analysis + previous_analysis) / 2)

    def mask_to_list_of_squares(self, mask: int):
        squares = []
        while mask != 0:
            squares.append(mask)
            mask -= chess.BB_SQUARES[mask.bit_length() - 1]
        return squares

    def game_phase(self, ply: int):
        if ply <= 14:
            return "opening"
        elif self.pieces_material <= self.endgame_point_material:
            return "endgame"
        else:
            return "mittelspiel"

    def mobility(self):
        # Mobility
        # Calculate all legal moves for white and black
        copy_board = self.board.copy()
        previous_main_mobility = self.main_mobility
        previous_enemy_mobility = self.enemy_mobility
        self.main_mobility = copy_board.legal_moves.count()
        # Change side to move by pushing a null move and calculate all moves for opponent
        copy_board.push(chess.Move.null())
        self.enemy_mobility = copy_board.legal_moves.count()

        self.mobility_increment = (
                                          self.main_mobility - previous_main_mobility) / self.all_material  # the extent to which my position has improved due to my move
        self.mobility_decrement = (
                                          previous_enemy_mobility - self.enemy_mobility) / self.all_material  # how much the enemy position improved due to my move
        return {'main_mobility': self.main_mobility, 'enemy_mobility': self.enemy_mobility}

    def control(self):
        attack_num = 0
        for pseudo_legal_move in self.board.pseudo_legal_moves:
            if self.board.is_capture(pseudo_legal_move):
                attack_num += 1
        return attack_num

    def king_control_zone(self, side):
        king_square = self.board.king(side)

        board = chess.Board.empty()
        board.kings = chess.BB_SQUARES[king_square]
        board.occupied_co[side] = chess.BB_SQUARES[king_square]
        board.occupied = board.occupied_co[side] + board.occupied_co[not side]

        if side == chess.WHITE:
            BB_KING_CONTROL_ZONE = board.attacks_mask(king_square) | board.attacks_mask(king_square) << 8 | \
                                   chess.BB_SQUARES[king_square]
        else:
            BB_KING_CONTROL_ZONE = board.attacks_mask(king_square) | board.attacks_mask(king_square) >> 8 | \
                                   chess.BB_SQUARES[king_square]
        return list(chess.SquareSet(BB_KING_CONTROL_ZONE))

    def attack_of_pieces(self, color, square):
        rank_pieces = chess.BB_RANK_MASKS[square] & self.board.occupied
        file_pieces = chess.BB_FILE_MASKS[square] & self.board.occupied
        diag_pieces = chess.BB_DIAG_MASKS[square] & self.board.occupied

        queens_and_rooks = self.board.queens | self.board.rooks
        queens_and_bishops = self.board.queens | self.board.bishops

        attackers = (
                (chess.BB_KING_ATTACKS[square] & self.board.kings) |
                (chess.BB_KNIGHT_ATTACKS[square] & self.board.knights) |
                (chess.BB_RANK_ATTACKS[square][rank_pieces] & queens_and_rooks) |
                (chess.BB_FILE_ATTACKS[square][file_pieces] & queens_and_rooks) |
                (chess.BB_DIAG_ATTACKS[square][
                     diag_pieces] & queens_and_bishops))  # |(chess.BB_PAWN_ATTACKS[not color][square] & board_.pawns))

        return attackers & self.board.occupied_co[color]

    def piece_attacks_mask(self, mask: int, side: bool, piece_sym: str):
        attacks_mask = 0
        attacks_count = 0
        attacking_pieces_count = 0
        king_square = self.board.king(side)
        KING_CONTROL_ZONE = self.BB_WHITE_KING_CONTROL_ZONE[king_square] if side == chess.WHITE \
            else self.BB_BLACK_KING_CONTROL_ZONE[king_square]
        if mask:
            for square in list(chess.SquareSet(mask)):
                attacks_mask += self.board.attacks_mask(square)
                attacks_mask &= KING_CONTROL_ZONE
                one_piece_attacks_count = self.count_bits(attacks_mask)
                if one_piece_attacks_count:
                    attacking_pieces_count += 1
                attacks_count += (one_piece_attacks_count * self.value_of_attacks_compliance[piece_sym])

        return [attacking_pieces_count, attacks_count]

    def count_bits(self, n):
        # from here https://stackoverflow.com/questions/9829578/fast-way-of-counting-non-zero-bits-in-positive-integer
        n = (n & 0x5555555555555555) + ((n & 0xAAAAAAAAAAAAAAAA) >> 1)
        n = (n & 0x3333333333333333) + ((n & 0xCCCCCCCCCCCCCCCC) >> 2)
        n = (n & 0x0F0F0F0F0F0F0F0F) + ((n & 0xF0F0F0F0F0F0F0F0) >> 4)
        n = (n & 0x00FF00FF00FF00FF) + ((n & 0xFF00FF00FF00FF00) >> 8)
        n = (n & 0x0000FFFF0000FFFF) + ((n & 0xFFFF0000FFFF0000) >> 16)
        n = (n & 0x00000000FFFFFFFF) + ((n & 0xFFFFFFFF00000000) >> 32)
        return n

    def pawn_is_attacked(self, square):
        piece_color = self.board.color_at(square)
        pawn_mask = chess.BB_PAWN_ATTACKS[not piece_color][square] & self.board.pawns & self.board.occupied_co[
            piece_color]
        if not pawn_mask:
            return True
        else:
            return False

    def king_control_zone_safety(self, side):  # https://www.chessprogramming.org/King_Safety#Attacking_King_Zone
        sum_attack_by_king_all_pieces = 0
        attacking_pieces_count = 0
        attack_weight = {0: 0, 1: 0, 2: 50, 3: 75, 4: 88, 5: 94, 6: 97, 7: 99, 8: 100, 9: 100, 10: 100, 11: 100,
                         12: 100}
        for piece_sym in self.value_of_attacks_compliance.keys():
            piece_mask = self.piece_mask_map()[piece_sym] & self.board.occupied_co[not side]
            piece_attack = self.piece_attacks_mask(piece_mask, side, piece_sym)
            attacking_pieces_count += piece_attack[0]
            sum_attack_by_king_all_pieces += piece_attack[1]

        return sum_attack_by_king_all_pieces * attack_weight[attacking_pieces_count] / 100

    def king_openness(self, side):
        copy_board = self.board.copy()
        king_square = self.board.king(side)
        copy_board.queens |= self.board.kings & self.board.occupied_co[side]
        copy_board.kings &= self.board.occupied_co[not side]
        return self.count_bits(copy_board.attacks_mask(square=king_square))

    def pseudo_legal_moves_one_piece(self, square, **kwargs):
        __board__ = kwargs.get('__board__', self.board)
        piece = __board__.piece_at(square)
        if piece is None:
            return 'no has piece in this square'
        copy_board = __board__.copy()
        copy_board.pawns &= __board__.occupied_co[not piece.color]
        copy_board.knights &= __board__.occupied_co[not piece.color]
        copy_board.bishops &= __board__.occupied_co[not piece.color]
        copy_board.rooks &= __board__.occupied_co[not piece.color]
        copy_board.queens &= __board__.occupied_co[not piece.color]
        copy_board.kings &= __board__.occupied_co[not piece.color]
        copy_board.turn = piece.color

        if piece.piece_type == 1:
            copy_board.pawns |= chess.BB_SQUARES[square]
        elif piece.piece_type == 2:
            copy_board.kings |= chess.BB_SQUARES[square]
        elif piece.piece_type == 3:
            copy_board.bishops |= chess.BB_SQUARES[square]
        elif piece.piece_type == 4:
            copy_board.rooks |= chess.BB_SQUARES[square]
        elif piece.piece_type == 5:
            copy_board.queens |= chess.BB_SQUARES[square]
        elif piece.piece_type == 6:
            copy_board.kings |= chess.BB_SQUARES[square]

        return copy_board.pseudo_legal_moves

    def bishop_activity(self, side, copy_board=None):
        if copy_board is None:
            copy_board = self.board.copy()

            copy_board.occupied_co[side] = (self.board.pawns | self.board.bishops) & self.board.occupied_co[side]
            copy_board.occupied_co[not side] = self.board.occupied_co[not side]
            copy_board.occupied = copy_board.occupied_co[0] | copy_board.occupied_co[1]

            copy_board.knights &= self.board.occupied_co[side]
            copy_board.rooks &= self.board.occupied_co[side]
            copy_board.queens &= self.board.occupied_co[side]
            copy_board.kings &= self.board.occupied_co[side]

            copy_board.turn = side

        mask = 0
        for square in chess.SquareSet(copy_board.bishops & copy_board.occupied_co[side]):
            mask |= copy_board.attacks_mask(square)
        pawn_main_color = mask & copy_board.pawns & copy_board.occupied_co[side]

        movable_pawns = list(
            filter(lambda sq: len(list(self.pseudo_legal_moves_one_piece(square=sq, __board__=copy_board))),
                   chess.SquareSet(pawn_main_color)))
        if not len(movable_pawns):
            return self.count_bits(mask)
        movable_pawns_mask = 0
        for sq in movable_pawns:
            movable_pawns_mask |= chess.BB_SQUARES[sq]
        copy_board.occupied_co[side] -= movable_pawns_mask
        copy_board.occupied = copy_board.occupied_co[0] | copy_board.occupied_co[1]
        copy_board.pawns -= movable_pawns_mask
        return self.bishop_activity(side, copy_board)

    def count_legal_moves_by_piece(self, side, piece_sym: str):
        activity_count = 0
        piece_mask = self.piece_mask_map()[piece_sym]
        for sq in chess.SquareSet(piece_mask & self.board.occupied_co[side]):
            activity_count += self.count_bits(self.board.attacks_mask(square=sq))
        return activity_count

    def pieces_usefulness_dict(self):
        side = self.board.turn

        if self.knights_num:
            knight_coeff = self.count_legal_moves_by_piece(side, 'N')
        else:
            knight_coeff = -1

        if self.bishops_num:
            bishop_coeff = self.bishop_activity(side)
        else:
            bishop_coeff = -1

        if self.rooks_num + self.queens_num:
            rook_and_queen_coeff = self.count_legal_moves_by_piece(side, 'R') + self.count_legal_moves_by_piece(side,
                                                                                                                'Q')
        else:
            rook_and_queen_coeff = -1
        return {"N": knight_coeff, "B": bishop_coeff, "R_Q": rook_and_queen_coeff}


class OnMovesInfo():
    def __init__(self, id_game: str, game: chess.pgn.Game, color: str):
        self.id_game = id_game
        self.game = game
        self.color = color
        self.moves_df = pd.DataFrame(columns=['id_game', 'move_number', 'white_move', 'black_move',
                                              'analysis', 'CP_loss', 'CP_loss_by_cauchy', 'pieces_material', 'pawns',
                                              'game_phase', 'mobility_increment', 'mobility_decrement',
                                              'control', 'king_safety', 'king_openness', 'knight_activity_coeff',
                                              'bishop_activity_coeff', 'rook_and_queen_activity_coeff'])
        self.moves_df.loc[0] = pd.Series()  # add first empty row
        self.add_new_row = True
        self.board = ModBoard()

        self.generate_moves_data()

    opposite = lambda self, color: 'white' if color == 'black' else 'black'

    color_index = lambda self, color: chess.WHITE if color == 'white' else chess.BLACK

    def generate_moves_data(self):
        color_index = self.color_index(self.color)
        for ply, move in enumerate(list(self.game.mainline_moves())):
            side = self.board.board.turn
            self.board.push_move(move)
            m_len = self.moves_df.index.size - 1  # length

            if ~ ply & 1 == color_index:
                # m_len = self.moves_df.index.size - 1  # length
                self.moves_df.at[m_len, 'id_game'] = self.id_game
                self.moves_df.at[m_len, 'move_number'] = int((ply + 2) / 2)
                self.moves_df.at[m_len, f'{self.color}_move'] = move.uci()
                # ------------------------------------------------------------------------------------------------
                self.board.comp_CP_analysis()
                self.moves_df.at[m_len, 'analysis'] = self.board.analysis
                self.moves_df.at[m_len, 'CP_loss'] = self.board.delta_CP
                self.moves_df.at[m_len, 'CP_loss_by_cauchy'] = self.board.delta_CP_by_cauchy
                material_info = self.board.material(side)
                self.moves_df.at[m_len, 'pieces_material'] = material_info['pieces_material']
                self.moves_df.at[m_len, 'pawns'] = material_info['pawns']
                self.moves_df.at[m_len, 'game_phase'] = self.board.game_phase(ply)
                self.board.mobility()
                self.moves_df.at[m_len, 'mobility_increment'] = self.board.mobility_increment
                self.moves_df.at[m_len, 'mobility_decrement'] = self.board.mobility_decrement
                self.moves_df.at[m_len, 'control'] = self.board.control()
                self.moves_df.at[m_len, 'king_safety'] = self.board.king_control_zone_safety(side)
                self.moves_df.at[m_len, 'king_openness'] = self.board.king_openness(side)
                pu_dict = self.board.pieces_usefulness_dict()
                self.moves_df.at[m_len, 'knight_activity_coeff'] = pu_dict['N']
                self.moves_df.at[m_len, 'bishop_activity_coeff'] = pu_dict['B']
                self.moves_df.at[m_len, 'rook_and_queen_activity_coeff'] = pu_dict['R_Q']

                if self.add_new_row is False:
                    self.moves_df.loc[m_len + 1] = pd.Series()
                    self.add_new_row = True
                elif self.add_new_row is True:
                    self.add_new_row = False

            else:

                self.enemy_move = move.uci()
                enemy_color = self.opposite(self.color)
                self.moves_df.at[m_len, f'{enemy_color}_move'] = self.enemy_move
                self.board.mobility()
                self.board.comp_CP_analysis()

                if self.add_new_row is False:
                    self.moves_df.loc[m_len + 1] = pd.Series()
                    self.add_new_row = True
                elif self.add_new_row is True:
                    self.add_new_row = False
