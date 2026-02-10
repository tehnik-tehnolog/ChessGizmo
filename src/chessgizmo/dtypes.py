from sqlalchemy.dialects.postgresql import TEXT, BIGINT, SMALLINT, BOOLEAN, REAL

# Data types for the users table
users_dtypes = {
    'username': TEXT,
    'num_games': BIGINT
}

# Data types for the games_info table
games_info_dtypes = {
    'id_game': TEXT,
    'status': TEXT,
    'id_player': TEXT,
    'id_opponent': TEXT,
    'main_color': BOOLEAN,
    'main_rating': SMALLINT,
    'enemy_rating': SMALLINT,
    'winner': TEXT,
    'opening_ACP': REAL,
    'mittelspiel_and_endgame_ACP': REAL,
    'is_there_endgame': BOOLEAN,
    'opening_STDPL': REAL,
    'mittelspiel_and_endgame_STDPL': REAL,
    'opening_ACP_by_cauchy': REAL,
    'mittelspiel_and_endgame_ACP_by_cauchy': REAL,
    'opening_STDPL_by_cauchy': REAL,
    'mittelspiel_and_endgame_STDPL_by_cauchy': REAL,
    'av_opening_mobility_inc': REAL,
    'av_mittelspiel_endgame_mobility_inc': REAL,
    'av_opening_mobility_dec': REAL,
    'av_mittelspiel_endgame_mobility_dec': REAL,
    'av_opening_king_safety': TEXT,
    'av_mittelspiel_endgame_king_safety': TEXT,
    'av_opening_king_openness': REAL,
    'av_mittelspiel_endgame_king_openness': REAL,
    'av_opening_knight_activ_coeff': REAL,
    'av_mittelspiel_endgame_knight_activ_coeff': REAL,
    'av_opening_bishop_activ_coeff': REAL,
    'av_mittelspiel_endgame_bishop_activ_coeff': REAL,
    'av_opening_rook_queen_activ_coeff': REAL,
    'av_mittelspiel_endgame_rook_queen_activ_coeff': REAL,
    'av_mittelspiel_control': REAL,
    'av_endgame_control': REAL,
    'av_opening_control': REAL,
    'outcome': REAL
}

# Data types for the games_by_moves table
games_by_moves_dtypes = {
    'id_game': TEXT,
    'move_number': SMALLINT,
    'white_move': TEXT,
    'black_move': TEXT,
    'analysis': SMALLINT,
    'CP_loss': SMALLINT,
    'CP_loss_by_cauchy': REAL,
    'pieces_material': SMALLINT,
    'pawns': SMALLINT,
    'game_phase': TEXT,
    'mobility_increment': REAL,
    'mobility_decrement': REAL,
    'control': SMALLINT,
    'king_safety': SMALLINT,
    'king_openness': SMALLINT,
    'knight_activity_coeff': SMALLINT,
    'bishop_activity_coeff': SMALLINT,
    'rook_and_queen_activity_coeff': SMALLINT,
    'main_color': BOOLEAN
}