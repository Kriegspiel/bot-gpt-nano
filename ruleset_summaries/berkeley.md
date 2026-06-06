# Berkeley

Play ordinary chess on the true board, but the opponent's pieces and moves are hidden from you. You see your own private FEN plus the public referee log and public material fields supplied in the user prompt. The server sees the true position and accepts only moves that are legal in that true position.

Move attempts continue until a legal move stands. If an attempted move is illegal in the true position, the referee announces the rejection publicly, but the rejected move itself is not revealed to the opponent. A legal move ends the turn.

After a legal move, public announcements may include a capture square, one or more check directions, checkmate, or stalemate. Captures reveal only the square of the captured unit, not the captured identity and not the capturing piece. Promotion, castling, and en passant are not separately announced; en passant is treated as an ordinary capture announcement. Berkeley has no pawn-capture count and no captured-pawn identity, so `pawns_captured` is intentionally absent. No `ask_any` action exists in this ruleset.
