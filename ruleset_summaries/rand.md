# RAND

Play ordinary chess on the true board, while the opponent's pieces and moves remain hidden. RAND is more talkative than Berkeley and Wild 16: public rebuffs, pawn-try source squares, typed captures, promotions, checkmate, and stalemate can all shape the public belief state.

- Referee response to illegal tries: RAND makes rejections public. Normal rebuffs, no answers, and special impossible-move rejections can appear in the referee log. The rejected move itself is only known when the scorecard perspective exposes it.
- Capture announcements: after a legal capture, RAND announces the capture square and whether the captured unit was a pawn or a non-pawn piece. Use `material.pawns_captured` when it appears.
- Check announcements: after a legal move gives check, the referee announces the check direction: rank, file, long diagonal, short diagonal, knight, or multiple directions when applicable. The checking piece square is not announced.
- Pawn-capture / Any? handling: before each turn, the referee announces the source squares of the mover's pawns that currently have legal capture tries. This reveals which pawns may capture, but not target squares, captured identities, or en passant status. If the player is in check, only pawn captures that legally answer the check are included. These announcements do not force a pawn capture. There is no `ask_any` action.
- Promotion announcements: promotions are announced in RAND, but not the promoted piece type and not the promotion square. Treat the promoted piece choice as hidden.
- Stalemate: the stalemated player loses in RAND. This is the special RAND rule; do not evaluate stalemate as an ordinary draw here.
