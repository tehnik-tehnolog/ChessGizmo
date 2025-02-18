UPDATE games_by_moves
SET 
    mobility_increment = IF(mobility_increment = -1, NULL, mobility_increment),
    mobility_decrement = IF(mobility_decrement = -1, NULL, mobility_decrement),
    control = IF(control = -1, NULL, control),
    king_safety = IF(king_safety = -1, NULL, king_safety),
    king_openness = IF(king_openness = -1, NULL, king_openness),
    knight_activity_coeff = IF(knight_activity_coeff = -1, NULL, knight_activity_coeff),
    bishop_activity_coeff = IF(bishop_activity_coeff = -1, NULL, bishop_activity_coeff),
    rook_and_queen_activity_coeff = IF(rook_and_queen_activity_coeff = -1, NULL, rook_and_queen_activity_coeff);
