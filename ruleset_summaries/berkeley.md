# Berkeley

Play ordinary chess on the true board, but the opponent's pieces and moves are hidden from you. You see your own private FEN plus the public referee log and public material fields supplied in the user prompt. The server sees the true position and accepts only moves that are legal in that true position.

- Referee response to illegal tries: move attempts continue until a legal move stands. If an attempted move is illegal in the true position, the referee announces a public rejection and the mover must try again. The rejected move itself is not revealed to the opponent unless it appears from that player's own scorecard perspective.
- Capture announcements: after a legal capture, Berkeley announces the capture square only. It does not reveal the captured identity, the capturing piece, or whether the captured unit was a pawn.
- Check announcements: after a legal move gives check, the referee announces the check direction: rank, file, long diagonal, short diagonal, knight, or multiple directions when applicable. The checking piece square is not announced.
- Pawn-capture / Any? handling: there is no `ask_any` action, no pawn-capture count, and no captured-pawn identity. The prompt intentionally omits `pawns_captured`.
- Promotion announcements: promotion is legal but not announced. The promotion piece is chosen privately by the mover; castling and en passant are also not separately announced.
- Stalemate: the referee may announce stalemate; Berkeley uses ordinary stalemate-as-draw handling, not RAND's stalemate-loss rule.
