UPDATE games_by_moves
SET 
    mobility_increment = NULLIF(mobility_increment, -1),
    mobility_decrement = NULLIF(mobility_decrement, -1),
    control = NULLIF(control, -1),
    king_safety = NULLIF(king_safety, -1),
    king_openness = NULLIF(king_openness, -1),
    knight_activity_coeff = NULLIF(knight_activity_coeff, -1),
    bishop_activity_coeff = NULLIF(bishop_activity_coeff, -1),
    rook_and_queen_activity_coeff = NULLIF(rook_and_queen_activity_coeff, -1);