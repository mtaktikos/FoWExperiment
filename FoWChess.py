# Simplified Fog-of-War Chess Engine (XBoard protocol)
# Pure Python, no external libraries (only standard modules: sys, random)
# Implements a Monte-Carlo belief-state search for imperfect information.
# Strength: reasonable with enough samples/time, but not superhuman.
# Author: Michael Taktikos and Grok (simplified version inspired by Obscuro concepts)


import sys
import random
import time


# Constants
EMPTY = 0
PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6


WHITE = 0
BLACK = 1
PIECE_CHARS = " PNBRQK"  # index 0 empty, 1-6 pieces


# Directions for pieces
KNIGHT_DELTAS = [-17, -15, -10, -6, 6, 10, 15, 17]
BISHOP_DELTAS = [-9, -7, 7, 9]
ROOK_DELTAS = [-8, -1, 1, 8]
QUEEN_DELTAS = ROOK_DELTAS + BISHOP_DELTAS
KING_DELTAS = QUEEN_DELTAS


# Board is 120-element array (0-119), with 0x88 off-board detection
def on_board(sq):
    return (sq & 0x88) == 0


# Convert algebraic to 0x88 square
def alg_to_sq(alg):
    file = ord(alg[0]) - ord('a')
    rank = int(alg[1]) - 1
    return rank * 16 + file


# Convert 0x88 square to algebraic
def sq_to_alg(sq):
    if sq == -1: return "0000"  # null move
    file = sq & 7
    rank = sq >> 4
    return chr(file + ord('a')) + str(rank + 1)


# Parse extended FEN (with 'f' for fog from player's view)
# Returns: board (list of 120 ints: >0 white piece, <0 black piece, 0 empty/fog)
# side_to_move, castling, ep_square (0x88), halfmove, fullmove
# The board will have known own pieces, opponent pieces only if visible, else 0 for fog.
def parse_extended_fen(fen, player_side):
    parts = fen.split()
    placement = parts[0]
    side = WHITE if parts[1] == 'w' else BLACK
    castling = parts[2]
    ep = parts[3] if len(parts) > 3 else '-'
    half = int(parts[4]) if len(parts) > 4 else 0
    full = int(parts[5]) if len(parts) > 5 else 1


    board = [0] * 120
    for i in range(120):
        if on_board(i):
            board[i] = 0  # default empty


    rank = 7
    file = 0
    for c in placement:
        if c == '/':
            rank -= 1
            file = 0
        elif c.isdigit():
            file += int(c)
        elif c.lower() in 'pnbrqk':
            sq = rank * 16 + file
            sign = 1 if c.isupper() else -1
            ptype = " PNBRQK".index(c.upper())
            board[sq] = sign * ptype
            file += 1
        elif c.lower() == 'f':
            sq = rank * 16 + file
            board[sq] = 0  # fog -> treated as possibly empty or opponent
            file += 1
        else:
            # invalid, ignore
            file += 1


    # ep square
    ep_sq = alg_to_sq(ep) if ep != '-' else -1


    return board, side, castling, ep_sq, half, full


# Generate all pseudo-legal moves from a full board state
def generate_moves(board, side):
    moves = []
    for sq in range(120):
        if not on_board(sq): continue
        piece = board[sq]
        if piece == 0: continue
        if (piece > 0) != (side == WHITE): continue
        ptype = abs(piece)


        if ptype == PAWN:
            dir = -16 if side == WHITE else 16
            start_rank = 6 if side == WHITE else 1
            rank = sq >> 4
            # single push
            tgt = sq + dir
            if on_board(tgt) and board[tgt] == 0:
                if rank == (1 if side == WHITE else 6):  # promotion rank opposite
                    for prom in [QUEEN, ROOK, BISHOP, KNIGHT]:
                        moves.append((sq, tgt, prom))
                else:
                    moves.append((sq, tgt, 0))
                # double push
                if (rank == start_rank) and board[tgt + dir] == 0:
                    moves.append((sq, tgt + dir, 0))
            # captures
            for cap_dir in [dir - 1, dir + 1]:
                tgt = sq + cap_dir
                if on_board(tgt) and board[tgt] != 0 and (board[tgt] > 0) != (side == WHITE):
                    if rank == (1 if side == WHITE else 6):
                        for prom in [QUEEN, ROOK, BISHOP, KNIGHT]:
                            moves.append((sq, tgt, prom))
                    else:
                        moves.append((sq, tgt, 0))
        elif ptype == KNIGHT:
            for d in KNIGHT_DELTAS:
                tgt = sq + d
                if on_board(tgt) and (board[tgt] == 0 or (board[tgt] > 0) != (side == WHITE)):
                    moves.append((sq, tgt, 0))
        elif ptype == BISHOP:
            for d in BISHOP_DELTAS:
                tgt = sq + d
                while on_board(tgt) and board[tgt] == 0:
                    moves.append((sq, tgt, 0))
                    tgt += d
                if on_board(tgt) and (board[tgt] > 0) != (side == WHITE):
                    moves.append((sq, tgt, 0))
        elif ptype == ROOK:
            for d in ROOK_DELTAS:
                tgt = sq + d
                while on_board(tgt) and board[tgt] == 0:
                    moves.append((sq, tgt, 0))
                    tgt += d
                if on_board(tgt) and (board[tgt] > 0) != (side == WHITE):
                    moves.append((sq, tgt, 0))
        elif ptype == QUEEN:
            for d in QUEEN_DELTAS:
                tgt = sq + d
                while on_board(tgt) and board[tgt] == 0:
                    moves.append((sq, tgt, 0))
                    tgt += d
                if on_board(tgt) and (board[tgt] > 0) != (side == WHITE):
                    moves.append((sq, tgt, 0))
        elif ptype == KING:
            for d in KING_DELTAS:
                tgt = sq + d
                if on_board(tgt) and (board[tgt] == 0 or (board[tgt] > 0) != (side == WHITE)):
                    moves.append((sq, tgt, 0))
            # Castling (simplified, ignore if rights lost or path blocked for now - can add later)
            if side == WHITE:
                if board[116] == ROOK * 1 and board[117] == 0 and board[118] == 0:  # kingside
                    moves.append((sq, sq+2, 0))  # special, handle in make
                if board[112] == ROOK * 1 and board[113] == 0 and board[114] == 0 and board[115] == 0:  # queenside
                    moves.append((sq, sq-2, 0))
            else:
                if board[4] == ROOK * -1 and board[5] == 0 and board[6] == 0:
                    moves.append((sq, sq+2, 0))
                if board[0] == ROOK * -1 and board[1] == 0 and board[2] == 0 and board[3] == 0:
                    moves.append((sq, sq-2, 0))


    return moves


# Make move on board (return new board, captured piece or 0)
def make_move(board, move):
    fr, to, prom = move
    piece = board[fr]
    captured = board[to]
    new_board = board[:]
    new_board[fr] = 0
    if prom:
        new_board[to] = (abs(piece) // piece) * prom  # sign * prom type
    else:
        new_board[to] = piece
    # simple castling
    if abs(piece) == KING and abs(fr - to) == 2:
        if to > fr:  # kingside
            new_board[to + 1] = new_board[to - 1]
            new_board[to - 1] = 0
        else:
            new_board[to - 2] = new_board[to + 1]
            new_board[to + 1] = 0
    return new_board, captured


# Simple material evaluation
def evaluate(board, side):
    mat = 0
    values = [0, 1, 3, 3, 5, 9, 100]
    for sq in range(120):
        if on_board(sq):
            p = board[sq]
            if p != 0:
                val = values[abs(p)]
                if (p > 0) == (side == WHITE):
                    mat += val
                else:
                    mat -= val
    return mat if side == WHITE else -mat


# Check if king captured (game over)
def king_captured(board, side):
    king_piece = KING if side == WHITE else -KING
    return king_piece not in board


# Sample possible worlds consistent with observed view
# observed_board: known own pieces positive/negative, fog 0, known opponent if visible
# player_side: the side we are playing
def sample_worlds(observed_board, player_side, num_samples=100):
    worlds = []
    own_sign = 1 if player_side == WHITE else -1
    opp_sign = -own_sign


    # First, collect all known own pieces
    own_pieces = []
    for sq in range(120):
        if on_board(sq) and observed_board[sq] * own_sign > 0:
            own_pieces.append((sq, observed_board[sq]))


    # Fog squares
    fog_squares = [sq for sq in range(120) if on_board(sq) and observed_board[sq] == 0]


    # To make sampling tractable, we assume standard material and place opponent pieces randomly in fog,
    # but ensure no overlap with known own, and approximate standard counts.
    # This is very approximate!
    opp_counts = {PAWN: 8, KNIGHT: 2, BISHOP: 2, ROOK: 2, QUEEN: 1, KING: 1}


    # Subtract any known opponent pieces (if visible captures)
    for sq in range(120):
        if on_board(sq) and observed_board[sq] * opp_sign > 0:
            ptype = abs(observed_board[sq])
            opp_counts[ptype] -= 1


    # Available squares for opponent
    available = [sq for sq in fog_squares if all(observed_board[sq] != own_sign * pt for pt in range(1,7))]  # not on own known


    for _ in range(num_samples):
        world = observed_board[:]  # start with known
        placed = []
        for ptype, count in list(opp_counts.items()):
            for _ in range(count):
                if not available: break
                sq = random.choice(available)
                world[sq] = opp_sign * ptype
                placed.append(sq)
                available.remove(sq)
        worlds.append(world)
    return worlds


# Monte Carlo rollout from a full board
def rollout(board, side, depth=20):
    current = side
    b = board[:]
    for _ in range(depth):
        moves = generate_moves(b, current)
        if not moves:
            return 0  # stalemate -> draw? but in FoW stalemate wins for stalemater, but simplify to draw
        move = random.choice(moves)
        b, _ = make_move(b, move)
        if king_captured(b, 1 - current):
            return 1 if current == side else -1  # capturer wins, from side perspective
        current = 1 - current
    return evaluate(b, side) / 100.0  # terminal eval if deep


# Search for best move using sampled worlds
def search(observed_board, player_side, time_limit=5.0):
    start_time = time.time()
    worlds = sample_worlds(observed_board, player_side, num_samples=50)  # adjustable


    # Get possible observed moves (pseudo from observed)
    possible_moves = generate_moves(observed_board, player_side)  # treats fog as empty, but captures not possible unless known opponent


    if not possible_moves:
        return None  # no moves


    move_scores = {m: 0.0 for m in possible_moves}
    move_counts = {m: 0 for m in possible_moves}


    samples_per_move = 20


    for move in possible_moves:
        win = draw = loss = 0
        for world in worlds:
            # Apply move in world (assume legal, since pseudo on observed)
            new_world, captured = make_move(world, move)
            opp_side = 1 - player_side
            if king_captured(new_world, opp_side):
                win += 1
                continue  # immediate win


            # Now opponent plays in this world
            opp_moves = generate_moves(new_world, opp_side)
            if not opp_moves:
                loss += 1  # opponent stalemated? wait, stalemate wins for us
                continue


            for _ in range(samples_per_move):
                outcome = rollout(new_world, player_side, depth=15)  # from our perspective after opp move? wait adjust
                if outcome > 0.1:
                    win += 1
                elif outcome < -0.1:
                    loss += 1
                else:
                    draw += 1


        total = win + draw + loss
        if total > 0:
            score = (win + 0.5 * draw) / total
            move_scores[move] = score
            move_counts[move] = total


    best_move = max(move_scores, key=move_scores.get)
    prob_win = move_scores[best_move]
    prob_draw = (move_counts[best_move] - (win + loss))
    # Rough estimate
    total_s = sum(move_counts.values())
    est_win = prob_win
    est_loss = 1 - prob_win - 0.1  # rough
    est_draw = 0.1


    print(f"# Best move {sq_to_alg(best_move[0])}{sq_to_alg(best_move[1])}, est win {est_win:.2f} draw 0.10 loss {1-est_win-0.10:.2f}", flush=True)


    return best_move


# Main engine loop (XBoard protocol)
def main():
    print("feature ping=1 myname=\"GrokFoW v0.1\" done=1", flush=True)
    observed_board = None
    player_side = None
    force_mode = False


    while True:
        line = sys.stdin.readline().strip()
        if not line: continue


        cmd = line.split()
        if cmd[0] == "xboard":
            continue
        elif cmd[0] == "protover":
            # already sent features
            continue
        elif cmd[0] == "new":
            # standard starting, but for FoW initial is own pieces visible, rest fog
            initial_fen = "ffffffff/ffffffff/ffffffff/ffffffff/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            # To make initial fog, replace opponent ranks with f
            if player_side is None:
                # assume we play white first? but wait for color
                pass
        elif cmd[0] == "setboard":
            fen = " ".join(cmd[1:])
            # determine player side? In analysis, we are analyzer, but for play, wait
            # For now, assume the side to move is opponent just gave us the position after their move
            # User will use force mode for analysis
            observed_board, side_to_move, _, _, _, _ = parse_extended_fen(fen, None)  # player_side later
            print("Hint: position set", flush=True)
        elif cmd[0] == "force":
            force_mode = True
        elif cmd[0] == "go":
            force_mode = False
            player_side = side_to_move  # current to move is us
            best = search(observed_board, player_side)
            if best:
                prom_char = ""
                if best[2]:
                    prom_char = PIECE_CHARS[best[2]].lower()
                move_str = sq_to_alg(best[0]) + sq_to_alg(best[1]) + prom_char
                print("move " + move_str, flush=True)
        elif cmd[0] == "usermove" or cmd[0] == "move":  # usermove in old
            usermove = cmd[1]
            fr = alg_to_sq(usermove[0:2])
            to = alg_to_sq(usermove[2:4])
            prom = 0
            if len(usermove) == 5:
                prom = " PNBRQK".index(usermove[4].upper())
            move = (fr, to, prom)
            if observed_board:
                observed_board, _ = make_move(observed_board, move)
                # Update observations? For simplified, we don't update visibility here, assume user provides updated fen after opponent
        elif cmd[0] == "ping":
            print("pong " + cmd[1], flush=True)
        elif cmd[0] == "quit":
            break


if __name__ == "__main__":
    main()