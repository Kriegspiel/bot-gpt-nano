# Berkeley + Any

Play ordinary chess on the true board, with the opponent's pieces and moves hidden. You see your own private FEN, public referee announcements, public material fields, possible actions, and the exact legal moves supplied in the user prompt. The server is the referee and accepts only moves that are legal in the true hidden position.

- Referee response to illegal tries: move attempts continue until a legal move stands. Illegal attempts are publicly announced as rejections and the mover must try again. The rejected move itself is not revealed to the opponent unless it appears from that player's own scorecard perspective.
- Capture announcements: after a legal capture, Berkeley + Any announces the capture square only. It does not reveal whether the captured unit was a pawn or piece, and it does not reveal the capturing piece.
- Check announcements: after a legal move gives check, the referee announces the check direction: rank, file, long diagonal, short diagonal, knight, or multiple directions when applicable. The checking piece square is not announced.
- Pawn-capture / Any? handling: when `ask_any` is available, it asks whether the side to move has at least one legal pawn capture. A positive answer obligates this turn to finish with a legal pawn capture. There is no pawn-capture count and no captured-pawn identity, so `pawns_captured` is intentionally absent.
- Promotion announcements: promotion is legal but not announced. The promotion piece is chosen privately by the mover; castling and en passant are also not separately announced.
- Stalemate: the referee may announce stalemate; Berkeley + Any uses ordinary stalemate-as-draw handling, not RAND's stalemate-loss rule.
