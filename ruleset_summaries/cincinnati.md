# Cincinnati

Play ordinary chess on the true board, but the opponent's pieces and moves are hidden. A player may make tries until one legal move stands. Illegal and Nonsense rejections are public referee information; the actual hidden opponent move or failed try is visible only when the user prompt already includes it from your own scorecard perspective.

Before a turn, the referee may publicly announce that the player to move has a pawn capture. This is binary information: at least one legal pawn capture exists, but the count, source squares, target squares, and captured identities are not given. A pawn-capture announcement does not force a pawn capture; it only tells you that trying one may be informative or legal. If no pawn capture was announced, a pawn-capture try can be Nonsense.

After a legal capture, Cincinnati announces the capture square and whether the captured unit was a pawn or a non-pawn piece. Use `material.pawns_captured` when it appears. Check directions are public. Promotion and castling are not announced. There is no `ask_any` action.
