# Berkeley + Any

Play ordinary chess on the true board, with the opponent's pieces and moves hidden. You see your own private FEN, public referee announcements, public material fields, possible actions, and the exact legal moves supplied in the user prompt. The server is the referee and accepts only moves that are legal in the true hidden position.

Move attempts continue until a legal move stands. Illegal attempts are publicly announced as rejections, but the rejected move itself is not revealed to the opponent. Legal moves may publicly announce a capture square, check direction, checkmate, or stalemate. Captures reveal only the square of the captured unit, not whether it was a pawn or piece. Promotion, castling, and en passant are not separately announced; en passant is treated as an ordinary capture announcement. Berkeley + Any has no pawn-capture count and no captured-pawn identity, so `pawns_captured` is intentionally absent.

When `ask_any` is available, it asks whether the side to move has at least one legal pawn capture. A positive answer obligates this turn to finish with a legal pawn capture.
