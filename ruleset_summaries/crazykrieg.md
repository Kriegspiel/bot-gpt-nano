# CrazyKrieg

CrazyKrieg combines Kriegspiel with Crazyhouse. The true board is hidden as usual, but both players' reserves are public. You may make an ordinary chess move or a legal reserve drop when the user prompt lists that action as an allowed move. The allowed move list is authoritative for both normal moves and drops.

- Referee response to illegal tries: illegal normal moves and illegal reserve drops are public rejections. The mover keeps trying until a legal move or legal drop stands. The attempted square is known only from your own perspective if the prompt includes it.
- Capture announcements: captures announce the square and the exact reserve identity of the captured unit: pawn, knight, bishop, rook, or queen. A promoted pawn enters reserve as a pawn and is announced as a pawn if captured. Public reserve fields are important for planning drops.
- Check announcements: after a legal move or legal drop gives check, the referee announces the check direction: rank, file, long diagonal, short diagonal, knight, or multiple directions when applicable. The checking piece or drop square is not announced.
- Pawn-capture / Any? handling: `ask_any` checks visible pawns for legal captures. After a positive answer, one pawn-capture try is required. If that required try is illegal, the player is released and may finish the turn with any legal move. Pawn drops are separate from pawn captures and are legal only when present in `allowed_moves`.
- Promotion announcements: promotion is legal but not announced when it happens, and the promoted piece type remains hidden. If a promoted pawn is later captured, it enters reserve and is announced as a pawn.
- Stalemate: the referee may announce stalemate; CrazyKrieg uses ordinary stalemate-as-draw handling, not RAND's stalemate-loss rule.
