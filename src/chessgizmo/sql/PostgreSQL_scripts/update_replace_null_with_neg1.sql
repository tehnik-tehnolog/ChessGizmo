UPDATE games_info
SET 
    av_opening_mobility_inc = COALESCE(av_opening_mobility_inc, -1),
    av_opening_mobility_dec = COALESCE(av_opening_mobility_dec, -1),
    av_opening_king_safety = COALESCE(av_opening_king_safety, -1),
    av_opening_king_openness = COALESCE(av_opening_king_openness, -1),
    av_opening_knight_activ_coeff = COALESCE(av_opening_knight_activ_coeff, -1),
    av_opening_bishop_activ_coeff = COALESCE(av_opening_bishop_activ_coeff, -1),
    av_opening_rook_queen_activ_coeff = COALESCE(av_opening_rook_queen_activ_coeff, -1),
    av_opening_control = COALESCE(av_opening_control, -1),
    av_mittelspiel_endgame_mobility_inc = COALESCE(av_mittelspiel_endgame_mobility_inc, -1),
    av_mittelspiel_endgame_mobility_dec = COALESCE(av_mittelspiel_endgame_mobility_dec, -1),
    av_mittelspiel_endgame_king_safety = COALESCE(av_mittelspiel_endgame_king_safety, -1),
    av_mittelspiel_endgame_king_openness = COALESCE(av_mittelspiel_endgame_king_openness, -1),
    av_mittelspiel_endgame_knight_activ_coeff = COALESCE(av_mittelspiel_endgame_knight_activ_coeff, -1),
    av_mittelspiel_endgame_bishop_activ_coeff = COALESCE(av_mittelspiel_endgame_bishop_activ_coeff, -1),
    av_mittelspiel_endgame_rook_queen_activ_coeff = COALESCE(av_mittelspiel_endgame_rook_queen_activ_coeff, -1),
    av_mittelspiel_control = COALESCE(av_mittelspiel_control, -1),
    av_endgame_control = COALESCE(av_endgame_control, -1);