# Cincinnati

Play ordinary chess on the true board, but the opponent's pieces and moves are hidden. A player may make tries until one legal move stands. Own pieces are official; guessed opposing pieces on a player's board have no official force.

- Referee response to illegal tries: illegal and Nonsense rejections are public referee information. The failed try is retracted and the player keeps trying until a legal move stands. The actual failed move is visible only when the prompt includes it from your own scorecard perspective.
- Capture announcements: after a legal capture, Cincinnati announces the capture square and whether the captured unit was a pawn or a non-pawn piece. Use `material.pawns_captured` when it appears.
- Check announcements: after a legal move gives check, the referee announces the check direction: rank, file, long diagonal, short diagonal, knight, or multiple directions when applicable. The checking piece square is not announced.
- Pawn-capture / Any? handling: before a turn, the referee may publicly announce that the player to move has a pawn capture. This is binary information: at least one legal pawn capture exists, but the count, source squares, target squares, and captured identities are not given. The announcement does not force a pawn capture. If no pawn capture was announced, a pawn-capture try can be Nonsense. There is no `ask_any` action.
- Promotion announcements: promotion is legal but not announced, and the promoted piece type remains hidden. Castling is also not announced.
- Stalemate: the referee may announce stalemate; Cincinnati uses ordinary stalemate-as-draw handling, not RAND's stalemate-loss rule.
