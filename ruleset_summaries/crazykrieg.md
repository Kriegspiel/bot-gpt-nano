# CrazyKrieg

CrazyKrieg combines Kriegspiel with Crazyhouse. The true board is hidden as usual, but both players' reserves are public. You may make an ordinary chess move or a legal reserve drop when the user prompt lists that action as an allowed move. The allowed move list is authoritative for both normal moves and drops.

A drop places a reserve piece on an empty true square. The opponent learns the dropped piece type from the public reserve change, but not the drop square unless later referee information reveals it. Pawn drops are not legal on the first or eighth rank. Drops can block check, give check, or give checkmate. Illegal move and drop attempts are public rejections, but the attempted square is known only from your own perspective if the prompt includes it.

Captures announce the square and the exact reserve identity of the captured unit: pawn, knight, bishop, rook, or queen. A promoted pawn enters reserve as a pawn and is announced as a pawn if captured. Public reserve fields are important for planning drops. Promotion is not announced when it happens. `ask_any` checks visible pawns for legal captures; after a positive answer, one pawn-capture try is required, and a failed required try releases the player to any legal move.
