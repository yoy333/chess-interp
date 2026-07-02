import chess


def has_passed_pawn(board: chess.Board) -> bool:
    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        for sq in pawns:
            f, r = chess.square_file(sq), chess.square_rank(sq)
            # Check for opposing pawns on same/adjacent files ahead
            enemy_color = not color
            stop_rank = 8 if color == chess.WHITE else -1
            step = 1 if color == chess.WHITE else -1

            is_passed = True
            for next_r in range(r + step, stop_rank, step):
                for next_f in [f-1, f, f+1]:
                    if 0 <= next_f <= 7:
                        target_sq = chess.square(next_f, next_r)
                        if board.piece_at(target_sq) == chess.Piece(chess.PAWN, enemy_color):
                            is_passed = False
                            break
                if not is_passed: break
            if is_passed: return True
    return False