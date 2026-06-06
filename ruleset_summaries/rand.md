# RAND

Play ordinary chess on the true board, while the opponent's pieces and moves remain hidden. RAND makes rejections public: normal rebuffs, no answers, and special impossible-move rejections can appear in the referee log. The rejected move itself is only known when the scorecard perspective exposes it.

Before each turn, the referee announces the source squares of the mover's pawns that currently have legal capture tries. This reveals which pawns may capture, but not the target squares, captured identities, or whether a capture is en passant. If the player is in check, only pawn captures that legally answer the check are included. These source-square announcements are information, not an obligation to capture.

Legal captures announce the capture square and whether the captured unit was a pawn or a non-pawn piece. Use `material.pawns_captured` when it appears. Check directions are public. RAND announces that a promotion happened, but not the promoted piece type or promotion square. Stalemate is a loss for the stalemated player. There is no `ask_any` action.
