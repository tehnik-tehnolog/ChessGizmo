from sqlalchemy import Integer, Boolean, SmallInteger, Float, VARCHAR, BIGINT
from sqlalchemy.dialects.mysql import TINYINT

chess_df_users_dtypes = {
    'username': VARCHAR(50),
    'num_games': Integer
}

chess_games_info_dtypes = {
    'id_game': VARCHAR(100),
    'status': VARCHAR(50),
    'id_player': VARCHAR(50),
    'id_opponent': VARCHAR(50),
    'main_color': Boolean,
    'outcome': Float,
    'main_rating': SmallInteger,
    'enemy_rating': SmallInteger,
    'opening_ACP': Float,
    'mittelspiel_and_endgame_ACP': Float,
    'is_there_endgame': Boolean,
    'opening_STDPL': Float,
    'mittelspiel_and_endgame_STDPL': Float,
    'opening_ACP_by_cauchy': Float,
    'mittelspiel_and_endgame_ACP_by_cauchy': Float,
    'opening_STDPL_by_cauchy': Float,
    'mittelspiel_and_endgame_STDPL_by_cauchy': Float
}

games_by_moves_dtypes = {
    'id_game': VARCHAR(100),
    'move_number': VARCHAR(6),
    'white_move': VARCHAR(6),
    'black_move': VARCHAR(6),
    'white_move_index': TINYINT,
    'black_move_index': TINYINT,
    'analysis': Float,
    'CP_loss': Float,
    'CP_loss_by_cauchy': Float,
    'pieces_material': SmallInteger,
    'pawns': TINYINT,
    'game_phase': VARCHAR(20),
    'mobility_increment': Float,
    'mobility_decrement': Float,
    'control': TINYINT,
    'king_safety': SmallInteger,
    'king_openness': TINYINT,
    'knight_activity_coeff': TINYINT,
    'bishop_activity_coeff': TINYINT,
    'rook_and_queen_activity_coeff': TINYINT
}


chess_df_users_dtypes_mod = {
    'username': VARCHAR(50),
    'num_games': Integer,
    'type_player': VARCHAR(50)
}

games_by_moves_dtypes_mod = {
    'id_game': VARCHAR(100),
    'move_number': BIGINT,
    'white_move': VARCHAR(6),
    'black_move': VARCHAR(6),
    'analysis': Float,
    'CP_loss': Float,
    'CP_loss_by_cauchy': Float,
    'pieces_material': SmallInteger,
    'pawns': TINYINT,
    'game_phase': VARCHAR(50),
    'mobility_increment': Float,
    'mobility_decrement': Float,
    'control': TINYINT,
    'king_safety': SmallInteger,
    'king_openness': TINYINT,
    'knight_activity_coeff': TINYINT,
    'bishop_activity_coeff': TINYINT,
    'rook_and_queen_activity_coeff': TINYINT,
    'main_color': VARCHAR(20)
}

chess_games_info_dtypes_mod = {
    'id_game': VARCHAR(100),
    'status': VARCHAR(50),
    'id_player': VARCHAR(50),
    'id_opponent': VARCHAR(50),
    'main_color': VARCHAR(50), # Boolean,
    'main_rating': SmallInteger,
    'enemy_rating': SmallInteger,
    'mittelspiel_and_endgame_ACP': Float,
    'is_there_endgame': Boolean,
    'opening_ACP': Float,
    'opening_STDPL': Float,
    'mittelspiel_and_endgame_STDPL': Float,
    'opening_ACP_by_cauchy': Float,
    'mittelspiel_and_endgame_ACP_by_cauchy': Float,
    'opening_STDPL_by_cauchy': Float,
    'mittelspiel_and_endgame_STDPL_by_cauchy': Float,
    'av_opening_mobility_inc': Float,
    'av_mittelspiel_endgame_mobility_inc': Float,
    'av_opening_mobility_dec': Float,
    'av_mittelspiel_endgame_mobility_dec': Float,
    'av_opening_king_safety': Float,
    'av_mittelspiel_endgame_king_safety': Float,
    'av_opening_king_openness': Float,
    'av_mittelspiel_endgame_king_openness': Float,
    'av_opening_knight_activ_coeff': Float,
    'av_mittelspiel_endgame_knight_activ_coeff': Float,
    'av_opening_bishop_activ_coeff': Float,
    'av_mittelspiel_endgame_bishop_activ_coeff': Float,
    'av_opening_rook_queen_activ_coeff': Float,
    'av_mittelspiel_endgame_rook_queen_activ_coeff': Float,
    'av_mittelspiel_control': Float,
    'av_endgame_control': Float,
    'av_opening_control': Float,
    'outcome': Float
}