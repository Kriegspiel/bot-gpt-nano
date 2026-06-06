# English

Play ordinary chess on the true board, with the opponent's pieces and moves hidden. Illegal moves are public referee announcements and must be retracted; the player keeps trying until a legal move stands. The prompt's recent scorecard turns carry the public announcements and any private move details the server allows from your own perspective.

After a legal capture, English rules announce that a capture occurred and the capture square, but not the capturing man and not the captured man. Ordinary captures do not reveal whether the captured unit was a pawn or piece, so `pawns_captured` is intentionally absent. En passant is the exception: the platform announces en passant explicitly on the capturing pawn's destination square. Check directions are public. Promotion and castling are not separately announced.

When `ask_any` is available, it asks whether any legal pawn capture exists. After a positive answer, the player must try one pawn capture. If that required pawn-capture try is illegal, the player is released and may finish the turn with any legal move.
