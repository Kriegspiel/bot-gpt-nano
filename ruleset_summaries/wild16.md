# Wild 16

Play ordinary chess on the true board, with the opponent's pieces and moves hidden. Wild 16 is the ICC-style server ruleset: failed attempts do not end the turn, and a legal attempt stands as the move.

- Referee response to illegal tries: illegal move attempts are private to the player who made them. Do not infer opponent illegal attempts from silence or missing public log entries. The mover keeps trying until a legal move stands.
- Capture announcements: after a legal capture, Wild 16 announces the captured square and whether the captured unit was a pawn or a non-pawn piece. En passant uses the captured pawn's square. Use `material.pawns_captured` when it appears.
- Check announcements: after a legal move gives check, the referee announces the check direction: rank, file, long diagonal, short diagonal, knight, or multiple directions when applicable. The checking piece square is not announced.
- Pawn-capture / Any? handling: before each turn, the referee publicly announces the exact count of legal pawn captures available to the player to move. The count does not reveal which pawn can capture, target squares, captured identities, or en passant status. The count does not force a pawn capture. There is no `ask_any` action.
- Promotion announcements: promotion is legal but not announced, and the promoted piece type remains hidden. Castling is also not announced.
- Stalemate: the referee may announce stalemate; Wild 16 uses ordinary stalemate-as-draw handling, not RAND's stalemate-loss rule.
