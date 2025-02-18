UPDATE chess_games_info cgi
LEFT JOIN (
    SELECT 
        id_game,
        AVG(mobility_increment) AS opening_mobility_inc_avg,
        AVG(mobility_decrement) AS opening_mobility_dec_avg,
        AVG(king_safety) AS opening_king_safety_avg,
        AVG(king_openness) AS opening_king_openness_avg,
        AVG(knight_activity_coeff) AS opening_knight_activ_coeff_avg,
        AVG(bishop_activity_coeff) AS opening_bishop_activ_coeff_avg,
        AVG(rook_and_queen_activity_coeff) AS opening_rook_queen_activ_coeff_avg,
        AVG(control) AS opening_control_avg
    FROM games_by_moves
    WHERE game_phase = 'opening'
    GROUP BY id_game
) opening ON cgi.id_game = opening.id_game
LEFT JOIN (
    SELECT 
        id_game,
        AVG(mobility_increment) AS mittel_end_mobility_inc_avg,
        AVG(mobility_decrement) AS mittel_end_mobility_dec_avg,
        AVG(king_safety) AS mittel_end_king_safety_avg,
        AVG(king_openness) AS mittel_end_king_openness_avg,
        AVG(knight_activity_coeff) AS mittel_end_knight_activ_coeff_avg,
        AVG(bishop_activity_coeff) AS mittel_end_bishop_activ_coeff_avg,
        AVG(rook_and_queen_activity_coeff) AS mittel_end_rook_queen_activ_coeff_avg
    FROM games_by_moves
    WHERE game_phase != 'opening'
    GROUP BY id_game
) mittel_end ON cgi.id_game = mittel_end.id_game
LEFT JOIN (
    SELECT 
        id_game,
        AVG(control) AS mittel_control_avg
    FROM games_by_moves
    WHERE game_phase = 'mittelspiel'
    GROUP BY id_game
) mittel ON cgi.id_game = mittel.id_game
LEFT JOIN (
    SELECT 
        id_game,
        AVG(control) AS end_control_avg
    FROM games_by_moves
    WHERE game_phase = 'endgame'
    GROUP BY id_game
) end ON cgi.id_game = end.id_game
SET 
    cgi.av_opening_mobility_inc = opening.opening_mobility_inc_avg,
    cgi.av_opening_mobility_dec = opening.opening_mobility_dec_avg,
    cgi.av_opening_king_safety = opening.opening_king_safety_avg,
    cgi.av_opening_king_openness = opening.opening_king_openness_avg,
    cgi.av_opening_knight_activ_coeff = opening.opening_knight_activ_coeff_avg,
    cgi.av_opening_bishop_activ_coeff = opening.opening_bishop_activ_coeff_avg,
    cgi.av_opening_rook_queen_activ_coeff = opening.opening_rook_queen_activ_coeff_avg,
    cgi.av_opening_control = opening.opening_control_avg,
    cgi.av_mittelspiel_endgame_mobility_inc = mittel_end.mittel_end_mobility_inc_avg,
    cgi.av_mittelspiel_endgame_mobility_dec = mittel_end.mittel_end_mobility_dec_avg,
    cgi.av_mittelspiel_endgame_king_safety = mittel_end.mittel_end_king_safety_avg,
    cgi.av_mittelspiel_endgame_king_openness = mittel_end.mittel_end_king_openness_avg,
    cgi.av_mittelspiel_endgame_knight_activ_coeff = mittel_end.mittel_end_knight_activ_coeff_avg,
    cgi.av_mittelspiel_endgame_bishop_activ_coeff = mittel_end.mittel_end_bishop_activ_coeff_avg,
    cgi.av_mittelspiel_endgame_rook_queen_activ_coeff = mittel_end.mittel_end_rook_queen_activ_coeff_avg,
    cgi.av_mittelspiel_control = mittel.mittel_control_avg,
    cgi.av_endgame_control = end.end_control_avg;