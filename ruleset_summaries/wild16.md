# Wild 16

Play ordinary chess on the true board, with the opponent's pieces and moves hidden. Wild 16 keeps illegal move attempts private to the player who made them, so opponent illegal attempts should not be inferred from silence or missing log entries. A failed attempt does not end the turn; the player keeps trying until a legal move stands.

Before each turn, the referee publicly announces the exact number of legal pawn captures available to the player to move, using the pawn-tries count. The count includes only pawn captures that are legal in the true position and that resolve check if the player is in check. The count does not reveal which pawn can capture, where the capture lands, what is captured, or whether the capture is en passant. The count does not force a pawn capture.

After a legal capture, Wild 16 announces the captured square and whether the captured unit was a pawn or a non-pawn piece. Use `material.pawns_captured` when it appears. Check directions are public. Promotion, castling, and en passant are not separately announced. There is no `ask_any` action.
