# English

Play ordinary chess on the true board, with the opponent's pieces and moves hidden. The prompt's recent scorecard turns carry the public announcements and any private move details the server allows from your own perspective.

- Referee response to illegal tries: illegal moves are public referee announcements and must be retracted. The player keeps trying until a legal move stands.
- Capture announcements: after a legal capture, English rules announce that a capture occurred and the capture square, but not the capturing man and not the captured man. Ordinary captures do not reveal whether the captured unit was a pawn or piece, so `pawns_captured` is intentionally absent. En passant is the exception: it is announced explicitly on the capturing pawn's destination square.
- Check announcements: after a legal move gives check, the referee announces the check direction: rank, file, long diagonal, short diagonal, knight, or multiple directions when applicable. The checking piece square is not announced.
- Pawn-capture / Any? handling: when `ask_any` is available, it asks whether any legal pawn capture exists. After a positive answer, the player must try one pawn capture. If that required pawn-capture try is illegal, the player is released and may finish the turn with any legal move.
- Promotion announcements: promotion is legal but not announced, and the promoted piece type remains hidden. Castling is also not announced.
- Stalemate: the referee may announce stalemate; English uses ordinary stalemate-as-draw handling, not RAND's stalemate-loss rule.
