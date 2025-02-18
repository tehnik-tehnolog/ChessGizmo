UPDATE chess_games_info
SET 
    av_opening_mobility_inc = IF(av_opening_mobility_inc IS NULL, -1, av_opening_mobility_inc),
    av_opening_mobility_dec = IF(av_opening_mobility_dec IS NULL, -1, av_opening_mobility_dec),
    av_opening_king_safety = IF(av_opening_king_safety IS NULL, -1, av_opening_king_safety),
    av_opening_king_openness = IF(av_opening_king_openness IS NULL, -1, av_opening_king_openness),
    av_opening_knight_activ_coeff = IF(av_opening_knight_activ_coeff IS NULL, -1, av_opening_knight_activ_coeff),
    av_opening_bishop_activ_coeff = IF(av_opening_bishop_activ_coeff IS NULL, -1, av_opening_bishop_activ_coeff),
    av_opening_rook_queen_activ_coeff = IF(av_opening_rook_queen_activ_coeff IS NULL, -1, av_opening_rook_queen_activ_coeff),
    av_opening_control = IF(av_opening_control IS NULL, -1, av_opening_control),
    av_mittelspiel_endgame_mobility_inc = IF(av_mittelspiel_endgame_mobility_inc IS NULL, -1, av_mittelspiel_endgame_mobility_inc),
    av_mittelspiel_endgame_mobility_dec = IF(av_mittelspiel_endgame_mobility_dec IS NULL, -1, av_mittelspiel_endgame_mobility_dec),
    av_mittelspiel_endgame_king_safety = IF(av_mittelspiel_endgame_king_safety IS NULL, -1, av_mittelspiel_endgame_king_safety),
    av_mittelspiel_endgame_king_openness = IF(av_mittelspiel_endgame_king_openness IS NULL, -1, av_mittelspiel_endgame_king_openness),
    av_mittelspiel_endgame_knight_activ_coeff = IF(av_mittelspiel_endgame_knight_activ_coeff IS NULL, -1, av_mittelspiel_endgame_knight_activ_coeff),
    av_mittelspiel_endgame_bishop_activ_coeff = IF(av_mittelspiel_endgame_bishop_activ_coeff IS NULL, -1, av_mittelspiel_endgame_bishop_activ_coeff),
    av_mittelspiel_endgame_rook_queen_activ_coeff = IF(av_mittelspiel_endgame_rook_queen_activ_coeff IS NULL, -1, av_mittelspiel_endgame_rook_queen_activ_coeff),
    av_mittelspiel_control = IF(av_mittelspiel_control IS NULL, -1, av_mittelspiel_control),
    av_endgame_control = IF(av_endgame_control IS NULL, -1, av_endgame_control);
