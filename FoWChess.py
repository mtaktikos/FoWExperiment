# Simplified Fog-of-War Chess Engine (XBoard protocol)
# Pure Python, no external libraries (only standard modules: sys, random)
# Implements a belief-state search (IS-PCFR+) for imperfect information.
# Algorithm: Predictive CFR+ (PCFR+) with Positive Regret Matching+ (PRM+) and
#            last-iterate play (no strategic averaging at runtime).
#            Converges to Nash equilibrium strategies in imperfect-information games
#            with faster last-iterate convergence than vanilla CFR.
# Based on: Farina, Kroer, Sandholm (2021) "Faster Game Solving via Predictive
#           Blackwell Approachability: Connecting Regret Matching and Mirror Descent";
#           Lanctot et al. (2009) "Monte Carlo Sampling for Regret Minimization in
#           Extensive Games", adapted from github.com/tansey/pycfr.
# Author: Michael Taktikos and Grok (simplified version inspired by Obscuro/Ludii concepts)


import sys
import random
import time


# Constants
EMPTY = 0
FOG = 99  # Special marker for fog squares
PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6


WHITE = 0
BLACK = 1
PIECE_CHARS = " PNBRQK"  # index 0 empty, 1-6 pieces


# Directions for pieces (0x88 board: rank*16 + file, so rank stride = 16)
KNIGHT_DELTAS = [-33, -31, -18, -14, 14, 18, 31, 33]
BISHOP_DELTAS = [-17, -15, 15, 17]
ROOK_DELTAS = [-16, -1, 1, 16]
QUEEN_DELTAS = ROOK_DELTAS + BISHOP_DELTAS
KING_DELTAS = QUEEN_DELTAS


# IS-CFR / Belief-state tuning constants
EVAL_NORMALIZATION_FACTOR = 200.0    # maps centipawn evaluation to [0, 1] range
BELIEF_MIN_SIMS_PER_WORLD = 10       # floor on CFR iterations per determinized world
BELIEF_MAX_SIMS_PER_WORLD = 900      # cap on CFR iterations per determinized world
BELIEF_TOTAL_SIM_BUDGET = 30000      # total simulation budget distributed across worlds
BELIEF_REFILL_THRESHOLD = 0.5        # re-sample when surviving worlds fall below this fraction
SAMPLING_ATTEMPT_MULTIPLIER = 15     # max_attempts = desired_worlds * this factor

# PCFR+-specific constants
CFR_TREE_DEPTH = 2         # levels of full tree expansion in External Sampling CFR:
                           #   depth 1 = explore all our immediate moves, then sample
                           #             opponent's reply and roll out from there.
CFR_ROLLOUT_DEPTH = 30     # random rollout depth at leaf nodes (same as former GRAVE)


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
# side_to_move, castling, ep_square (0x88), halfmove, fullmove, white_in_hand, black_in_hand
# The board will have known own pieces, opponent pieces only if visible, else 0 for fog.
# Pieces in hand are stored as dicts {piece_type: count}
def parse_extended_fen(fen, player_side):
    parts = fen.split()
    placement = parts[0]
    
    # Parse pieces in hand from placement (format: ranks[pieces])
    # Uppercase = White pieces captured (by Black), go to Black's hand
    # Lowercase = Black pieces captured (by White), go to White's hand
    white_in_hand = {PAWN: 0, KNIGHT: 0, BISHOP: 0, ROOK: 0, QUEEN: 0, KING: 0}
    black_in_hand = {PAWN: 0, KNIGHT: 0, BISHOP: 0, ROOK: 0, QUEEN: 0, KING: 0}
    
    # Check if there are brackets in the placement
    if '[' in placement:
        # Split board part from pieces in hand
        bracket_start = placement.index('[')
        board_part = placement[:bracket_start]
        hand_part = placement[bracket_start:]
        
        # Parse pieces in hand
        # Format: [pieces] where case indicates which color was captured
        in_bracket = False
        for c in hand_part:
            if c == '[':
                in_bracket = True
            elif c == ']':
                in_bracket = False
            elif in_bracket and c.upper() in 'PNBRQK':
                ptype = " PNBRQK".index(c.upper())
                if c.isupper():  # White piece captured -> goes to Black's hand
                    black_in_hand[ptype] += 1
                else:  # Black piece captured -> goes to White's hand
                    white_in_hand[ptype] += 1
        
        placement = board_part
    
    side = BLACK if len(parts) > 1 and parts[1] == 'b' else WHITE
    castling = parts[2] if len(parts) > 2 else '-'
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
            board[sq] = FOG  # fog -> treated as possibly empty or opponent
            file += 1
        else:
            # invalid, ignore
            file += 1


    # ep square
    ep_sq = alg_to_sq(ep) if ep != '-' else -1


    return board, side, castling, ep_sq, half, full, white_in_hand, black_in_hand


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
            dir = 16 if side == WHITE else -16
            start_rank = 1 if side == WHITE else 6
            rank = sq >> 4
            # single push
            tgt = sq + dir
            if on_board(tgt) and (board[tgt] == 0 or board[tgt] == FOG):
                if rank == (6 if side == WHITE else 1):  # promotion rank (rank 7 for white, rank 2 for black)
                    for prom in [QUEEN, ROOK, BISHOP, KNIGHT]:
                        moves.append((sq, tgt, prom))
                else:
                    moves.append((sq, tgt, 0))
                # double push
                if (rank == start_rank) and (board[tgt + dir] == 0 or board[tgt + dir] == FOG):
                    moves.append((sq, tgt + dir, 0))
            # captures
            for cap_dir in [dir - 1, dir + 1]:
                tgt = sq + cap_dir
                if on_board(tgt) and board[tgt] != 0 and board[tgt] != FOG and (board[tgt] > 0) != (side == WHITE):
                    if rank == (6 if side == WHITE else 1):
                        for prom in [QUEEN, ROOK, BISHOP, KNIGHT]:
                            moves.append((sq, tgt, prom))
                    else:
                        moves.append((sq, tgt, 0))
        elif ptype == KNIGHT:
            for d in KNIGHT_DELTAS:
                tgt = sq + d
                if on_board(tgt) and (board[tgt] == 0 or board[tgt] == FOG or (board[tgt] != 0 and board[tgt] != FOG and (board[tgt] > 0) != (side == WHITE))):
                    moves.append((sq, tgt, 0))
        elif ptype == BISHOP:
            for d in BISHOP_DELTAS:
                tgt = sq + d
                while on_board(tgt) and (board[tgt] == 0 or board[tgt] == FOG):
                    moves.append((sq, tgt, 0))
                    tgt += d
                if on_board(tgt) and board[tgt] != 0 and board[tgt] != FOG and (board[tgt] > 0) != (side == WHITE):
                    moves.append((sq, tgt, 0))
        elif ptype == ROOK:
            for d in ROOK_DELTAS:
                tgt = sq + d
                while on_board(tgt) and (board[tgt] == 0 or board[tgt] == FOG):
                    moves.append((sq, tgt, 0))
                    tgt += d
                if on_board(tgt) and board[tgt] != 0 and board[tgt] != FOG and (board[tgt] > 0) != (side == WHITE):
                    moves.append((sq, tgt, 0))
        elif ptype == QUEEN:
            for d in QUEEN_DELTAS:
                tgt = sq + d
                while on_board(tgt) and (board[tgt] == 0 or board[tgt] == FOG):
                    moves.append((sq, tgt, 0))
                    tgt += d
                if on_board(tgt) and board[tgt] != 0 and board[tgt] != FOG and (board[tgt] > 0) != (side == WHITE):
                    moves.append((sq, tgt, 0))
        elif ptype == KING:
            for d in KING_DELTAS:
                tgt = sq + d
                if on_board(tgt) and (board[tgt] == 0 or board[tgt] == FOG or (board[tgt] != 0 and board[tgt] != FOG and (board[tgt] > 0) != (side == WHITE))):
                    moves.append((sq, tgt, 0))
            # Castling (simplified, ignore if rights lost or path blocked for now - can add later)
            if side == WHITE:
                if board[7] == ROOK * 1 and (board[5] == 0 or board[5] == FOG) and (board[6] == 0 or board[6] == FOG):  # kingside
                    moves.append((sq, sq+2, 0))  # special, handle in make
                if board[0] == ROOK * 1 and (board[1] == 0 or board[1] == FOG) and (board[2] == 0 or board[2] == FOG) and (board[3] == 0 or board[3] == FOG):  # queenside
                    moves.append((sq, sq-2, 0))
            else:
                if board[119] == ROOK * -1 and (board[117] == 0 or board[117] == FOG) and (board[118] == 0 or board[118] == FOG):
                    moves.append((sq, sq+2, 0))
                if board[112] == ROOK * -1 and (board[113] == 0 or board[113] == FOG) and (board[114] == 0 or board[114] == FOG) and (board[115] == 0 or board[115] == FOG):
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
        if to > fr:  # kingside: rook goes from to+1 (h-file) to to-1 (f-file)
            new_board[to - 1] = new_board[to + 1]
            new_board[to + 1] = 0
        else:  # queenside: rook goes from to-2 (a-file) to to+1 (d-file)
            new_board[to + 1] = new_board[to - 2]
            new_board[to - 2] = 0
    return new_board, captured


# Simple material evaluation
def evaluate(board, side):
    mat = 0
    values = [0, 1, 3, 3, 5, 9, 100]
    for sq in range(120):
        if on_board(sq):
            p = board[sq]
            if p != 0 and p != FOG:
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
# white_in_hand, black_in_hand: pieces captured by each side
def sample_worlds(observed_board, player_side, white_in_hand, black_in_hand, num_samples=100):
    worlds = []
    own_sign = 1 if player_side == WHITE else -1
    opp_sign = -own_sign

    # First, collect all known own pieces
    own_pieces = []
    for sq in range(120):
        if on_board(sq) and observed_board[sq] * own_sign > 0:
            own_pieces.append((sq, observed_board[sq]))

    # Fog squares
    fog_squares = [sq for sq in range(120) if on_board(sq) and (observed_board[sq] == 0 or observed_board[sq] == FOG)]

    # Calculate opponent pieces remaining using pieces in hand information
    # Standard material: 8 pawns, 2 knights, 2 bishops, 2 rooks, 1 queen, 1 king
    # Opponent pieces remaining = standard - (on_board + in_hand)
    standard_material = {PAWN: 8, KNIGHT: 2, BISHOP: 2, ROOK: 2, QUEEN: 1, KING: 1}
    
    # Determine which pieces we captured from opponent
    our_hand = white_in_hand if player_side == WHITE else black_in_hand
    
    # Count opponent pieces on board (visible ones)
    opp_on_board = {PAWN: 0, KNIGHT: 0, BISHOP: 0, ROOK: 0, QUEEN: 0, KING: 0}
    for sq in range(120):
        if on_board(sq) and observed_board[sq] * opp_sign > 0:
            ptype = abs(observed_board[sq])
            opp_on_board[ptype] += 1
    
    # Opponent pieces remaining to place in fog = standard - (on_board + captured_by_us)
    opp_counts = {}
    for ptype in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]:
        remaining = standard_material[ptype] - opp_on_board[ptype] - our_hand[ptype]
        opp_counts[ptype] = max(0, remaining)  # Can't be negative

    # Available squares for opponent
    available = [sq for sq in fog_squares if all(observed_board[sq] != own_sign * pt for pt in range(1,7))]  # not on own known

    for _ in range(num_samples):
        world = observed_board[:]  # start with known
        placed = []
        avail_copy = available[:]  # make a copy for this sample
        for ptype, count in list(opp_counts.items()):
            for _ in range(count):
                if not avail_copy: break
                sq = random.choice(avail_copy)
                world[sq] = opp_sign * ptype
                placed.append(sq)
                avail_copy.remove(sq)
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
def search(observed_board, player_side, white_in_hand, black_in_hand, time_limit=5.0):
    start_time = time.time()
    worlds = sample_worlds(observed_board, player_side, white_in_hand, black_in_hand, num_samples=50)  # adjustable


    # Get possible observed moves (pseudo from observed)
    possible_moves = generate_moves(observed_board, player_side)  # treats fog as empty, but captures not possible unless known opponent


    if not possible_moves:
        return None  # no moves


    move_scores = {m: 0.0 for m in possible_moves}
    move_counts = {m: 0 for m in possible_moves}


    samples_per_move = 20


    for move in possible_moves:
        move_win = move_draw = move_loss = 0
        for world in worlds:
            # Apply move in world (assume legal, since pseudo on observed)
            new_world, captured = make_move(world, move)
            opp_side = 1 - player_side
            if king_captured(new_world, opp_side):
                move_win += 1
                continue  # immediate win


            # Now opponent plays in this world
            opp_moves = generate_moves(new_world, opp_side)
            if not opp_moves:
                move_loss += 1  # opponent stalemated? wait, stalemate wins for us
                continue


            for _ in range(samples_per_move):
                outcome = rollout(new_world, player_side, depth=15)  # from our perspective after opp move? wait adjust
                if outcome > 0.1:
                    move_win += 1
                elif outcome < -0.1:
                    move_loss += 1
                else:
                    move_draw += 1


        total = move_win + move_draw + move_loss
        if total > 0:
            score = (move_win + 0.5 * move_draw) / total
            move_scores[move] = score
            move_counts[move] = total


    best_move = max(move_scores, key=move_scores.get)
    # Rough estimate of probabilities
    est_win = move_scores[best_move]
    est_loss = 1 - est_win - 0.1  # rough
    est_draw = 0.1


    print(f"# Best move {sq_to_alg(best_move[0])}{sq_to_alg(best_move[1])}, est win {est_win:.2f} draw 0.10 loss {1-est_win-0.10:.2f}", flush=True)


    return best_move


# ============================================================
# Belief State and IS-MCTS Implementation
# Inspired by Sandholm's Obscuro concepts for FoW Chess
# ============================================================


class ObservationHistory:
    """Tracks the sequence of fog-of-war FEN observations and the moves that followed them."""

    def __init__(self, player_side):
        self.player_side = player_side
        # Each entry: {'fog_fen': str, 'move': tuple or None}
        self.entries = []

    def add_observation(self, fog_fen):
        self.entries.append({'fog_fen': fog_fen, 'move': None})

    def record_move(self, move):
        """Attach a move to the most recent observation."""
        if self.entries:
            self.entries[-1]['move'] = move

    def latest_fog_fen(self):
        return self.entries[-1]['fog_fen'] if self.entries else None


def _is_world_consistent(world, observed_board, player_side):
    """
    Return True if *world* (a fully-specified board) is compatible with
    *observed_board* (which may contain FOG markers):

    * Squares where observed_board[sq] != FOG must match world[sq] exactly.
    * FOG squares may contain any opponent piece or be empty, but NOT our
      own pieces (those would be visible to us).
    """
    own_sign = 1 if player_side == WHITE else -1
    for sq in range(120):
        if not on_board(sq):
            continue
        obs = observed_board[sq]
        wld = world[sq]
        if obs == FOG:
            # Our own piece must never hide in fog
            if wld * own_sign > 0:
                return False
        else:
            if obs != wld:
                return False
    return True


def enumerate_worlds(observed_board, player_side, white_in_hand, black_in_hand,
                     max_worlds=200):
    """
    Sample up to *max_worlds* complete board states (determinizations) that
    are consistent with *observed_board*.

    Strategy:
      1. Identify FOG squares – the only candidates for hidden opponent pieces.
      2. Compute how many of each opponent piece type still remain unaccounted
         for (standard starting count minus visible board count minus captured).
      3. Randomly assign those remaining pieces to FOG squares.
      4. Verify consistency and deduplicate.
    """
    own_sign = 1 if player_side == WHITE else -1
    opp_sign = -own_sign

    standard_material = {PAWN: 8, KNIGHT: 2, BISHOP: 2, ROOK: 2, QUEEN: 1, KING: 1}

    # Count opponent pieces that are directly visible on the observed board
    opp_on_board = {pt: 0 for pt in standard_material}
    for sq in range(120):
        if on_board(sq) and observed_board[sq] * opp_sign > 0:
            ptype = abs(observed_board[sq])
            if ptype in opp_on_board:
                opp_on_board[ptype] += 1

    # Pieces we captured from the opponent (they sit in our hand)
    our_hand = white_in_hand if player_side == WHITE else black_in_hand

    # Build the list of opponent piece types that still need placing in fog
    opp_remaining = []
    for ptype in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]:
        count = standard_material[ptype] - opp_on_board[ptype] - our_hand[ptype]
        opp_remaining.extend([ptype] * max(0, count))

    # Only FOG-marked squares are candidates (squares we genuinely cannot see)
    fog_sqs = [sq for sq in range(120) if on_board(sq) and observed_board[sq] == FOG]

    worlds = []
    seen_hashes = set()
    max_attempts = max_worlds * SAMPLING_ATTEMPT_MULTIPLIER

    for _ in range(max_attempts):
        if len(worlds) >= max_worlds:
            break
        if opp_remaining and not fog_sqs:
            break
        if len(opp_remaining) > len(fog_sqs):
            # More pieces than available fog squares – give up sampling
            break

        # Build world: copy observed board, replace FOG squares with 0 first
        world = observed_board[:]
        for sq in fog_sqs:
            world[sq] = 0

        # Randomly place remaining opponent pieces into fog squares
        avail = fog_sqs[:]
        random.shuffle(avail)
        for ptype in opp_remaining:
            sq = avail.pop()
            world[sq] = opp_sign * ptype

        # Verify consistency (guards against edge cases after move propagation)
        if not _is_world_consistent(world, observed_board, player_side):
            continue

        key = tuple(world)
        if key in seen_hashes:
            continue
        seen_hashes.add(key)
        worlds.append(world)

    # Fallback to the original sampler if enumeration yielded nothing
    if not worlds:
        worlds = sample_worlds(observed_board, player_side,
                               white_in_hand, black_in_hand, max_worlds)
    return worlds


class BeliefState:
    """
    Maintains a set P of consistent positions (the belief state) for FoW chess.

    After each new fog_fen observation the set is updated:
      * Existing worlds that contradict the new observation are discarded.
      * If too few worlds remain, fresh ones are sampled by enumerate_worlds.

    After each move (ours or opponent's) every world in P is advanced by that
    move so the set stays in sync with the current game state.
    """

    def __init__(self, player_side, max_worlds=100):
        self.player_side = player_side
        self.max_worlds = max_worlds
        self.observation_history = ObservationHistory(player_side)
        self.worlds = []          # list of fully-specified board states
        self._last_wih = {pt: 0 for pt in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]}
        self._last_bih = {pt: 0 for pt in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_observation(self, fog_fen):
        """
        Incorporate a new fog_fen observation.

        Returns the parsed observed board, white_in_hand, black_in_hand so
        callers do not have to parse the FEN a second time.
        """
        self.observation_history.add_observation(fog_fen)
        board, _side, _cast, _ep, _half, _full, wih, bih = parse_extended_fen(
            fog_fen, self.player_side)
        self._last_wih = wih
        self._last_bih = bih

        # Keep worlds that remain consistent with the new observation
        consistent = [w for w in self.worlds
                      if _is_world_consistent(w, board, self.player_side)]

        refill_threshold = int(self.max_worlds * BELIEF_REFILL_THRESHOLD)
        if len(consistent) >= refill_threshold:
            self.worlds = consistent[:self.max_worlds]
        else:
            # Re-sample and merge with surviving worlds
            fresh = enumerate_worlds(board, self.player_side, wih, bih,
                                     self.max_worlds)
            combined = consistent + fresh
            seen = set()
            deduped = []
            for w in combined:
                k = tuple(w)
                if k not in seen:
                    seen.add(k)
                    deduped.append(w)
            self.worlds = deduped[:self.max_worlds]

        return board, wih, bih

    def propagate_our_move(self, move):
        """Advance every world through *move* (our move) and record it."""
        self.observation_history.record_move(move)
        self.worlds = self._apply_move_to_all(move)

    def propagate_opponent_move(self, move):
        """Advance every world through the opponent's *move*."""
        self.worlds = self._apply_move_to_all(move)

    def sample(self, n=None):
        """Return up to *n* worlds (all if n is None or exceeds pool size)."""
        if not self.worlds:
            return []
        if n is None or n >= len(self.worlds):
            return list(self.worlds)
        return random.sample(self.worlds, n)

    def size(self):
        return len(self.worlds)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_move_to_all(self, move):
        new_worlds = []
        for world in self.worlds:
            try:
                new_world, _ = make_move(world, move)
                new_worlds.append(new_world)
            except Exception:
                # Keep the world unchanged if make_move raises (e.g. illegal
                # move in this particular determinization).  The world will be
                # filtered out on the next observation update if it becomes
                # inconsistent with reality.
                new_worlds.append(world)
        return new_worlds


# ============================================================
# Random rollout used by CFR for leaf-node evaluation
# ============================================================


def random_rollout(board, side, root_side, depth=20):
    """
    Random playout from *board* with *side* to move.

    Returns (result, played) where:
      result – float in [0, 1] from *root_side*'s perspective
      played – list of (move, side) pairs in play order
    """
    current = side
    b = board[:]
    played = []
    for _ in range(depth):
        moves = generate_moves(b, current)
        if not moves:
            return 0.5, played          # stalemate → draw
        move = random.choice(moves)
        played.append((move, current))
        b, _ = make_move(b, move)
        if king_captured(b, 1 - current):
            result = 1.0 if current == root_side else 0.0
            return result, played
        current = 1 - current
    # Depth exhausted – material heuristic normalised to [0, 1]
    val = evaluate(b, root_side)
    return max(0.0, min(1.0, 0.5 + val / EVAL_NORMALIZATION_FACTOR)), played


# ============================================================
# PCFR+ (Predictive CFR+) for FoW Chess
# IS-PCFR+ – Information-Set External Sampling Monte Carlo PCFR+
#
# Based on: Farina, Kroer, Sandholm (2021) "Faster Game Solving via Predictive
#           Blackwell Approachability: Connecting Regret Matching and Mirror
#           Descent"; Lanctot et al. (2009) MCCFR.
#
# Key improvements over vanilla CFR:
#   1. PRM+ (Positive Regret Matching+): cumulative regrets are clamped to ≥ 0
#      after each update, eliminating harmful negative regret accumulation.
#   2. Predictive strategy: the strategy at iteration t uses
#        σ(t)[a] = RM+(R+(t-1)[a] + r(t-1)[a])
#      where r(t-1) is the *immediate* counterfactual regret from the previous
#      iteration (the "prediction").  This accelerates convergence.
#   3. Last-iterate play: at runtime we play the *current* strategy σ(t),
#      not the time-average.  PCFR+ last iterates converge to Nash equilibrium.
#
# Algorithm: External Sampling MCCFR with alternating player updates.
#   - At the traverser's nodes  → explore ALL legal moves (exact CF values).
#   - At the opponent's nodes   → sample ONE move from their current strategy.
#   - At depth limit / terminal → evaluate via a random rollout.
#
# Key data per tree node:
#   regret_sum[a]           – cumulative non-negative counterfactual regret (R+).
#   last_immediate_regret[a]– immediate regret r(t-1) used as prediction for
#                             the next iteration's strategy.
# ============================================================


class CFRNode:
    """
    Node for External Sampling MCCFR with PCFR+ updates.

    regret_sum[move]            – cumulative non-negative regrets (RM+ clamped).
    last_immediate_regret[move] – immediate regret from the most recent traversal
                                  iteration; used as the prediction in PCFR+.
    """

    __slots__ = ('board', 'side', 'parent', 'move', 'children', 'visits',
                 'all_moves', 'regret_sum', 'last_immediate_regret')

    def __init__(self, board, side, parent=None, move=None):
        self.board = board
        self.side = side          # side to move at this node
        self.parent = parent
        self.move = move          # move that led here from parent
        self.children = {}        # move -> CFRNode
        self.visits = 0
        moves = generate_moves(board, side)
        random.shuffle(moves)
        self.all_moves = moves
        self.regret_sum = {m: 0.0 for m in moves}           # cumulative R+, always ≥ 0
        self.last_immediate_regret = {m: 0.0 for m in moves} # immediate regret, prediction

    def is_terminal(self):
        return king_captured(self.board, WHITE) or king_captured(self.board, BLACK)

    def get_strategy(self):
        """
        Predictive RM+: σ(a) = max(R+(a) + r_pred(a), 0) / Σ max(R+(a) + r_pred(a), 0).

        The prediction r_pred is the immediate regret from the previous iteration.
        Falls back to a uniform distribution when no positive predictive regret exists.
        """
        moves = self.all_moves
        if not moves:
            return {}
        pos = {m: max(0.0, self.regret_sum.get(m, 0.0)
                      + self.last_immediate_regret.get(m, 0.0))
               for m in moves}
        total = sum(pos.values())
        if total > 0:
            return {m: pos[m] / total for m in moves}
        return {m: 1.0 / len(moves) for m in moves}


def _cfr_traverse(node, player_side, traverser, depth):
    """
    External Sampling MCCFR traversal with PCFR+ updates for one player (the *traverser*).

    At the traverser's nodes  → explore ALL legal moves, compute exact
                                counterfactual values, update regrets (PCFR+).
    At the opponent's nodes   → sample ONE move from their current strategy.
    At depth 0 / terminal     → evaluate with a random rollout.

    Always returns the game value from *player_side*'s perspective.
    """
    # ---- Terminal ----
    if node.is_terminal():
        return 1.0 if king_captured(node.board, 1 - player_side) else 0.0

    moves = node.all_moves
    if not moves:
        return 0.5  # stalemate

    # ---- Depth limit: random rollout ----
    if depth == 0:
        result, _ = random_rollout(node.board, node.side, player_side,
                                   CFR_ROLLOUT_DEPTH)
        return result

    strategy = node.get_strategy()

    if node.side == traverser:
        # ---- Traverser's node: explore ALL moves (exact counterfactual values) ----
        action_values = {}
        for move in moves:
            if move not in node.children:
                new_board, _ = make_move(node.board, move)
                child = CFRNode(new_board, 1 - node.side, parent=node, move=move)
                node.children[move] = child
            action_values[move] = _cfr_traverse(
                node.children[move], player_side, traverser, depth - 1)

        # Node value = expectation under current strategy (player_side's perspective)
        node_value = sum(strategy.get(m, 0.0) * action_values[m] for m in moves)

        # ---- PCFR+ regret update ----
        # Compute immediate (per-iteration) counterfactual regret, store as the
        # prediction for the NEXT iteration, then apply RM+ clamping (R+ ≥ 0).
        sign = 1.0 if traverser == player_side else -1.0
        for m in moves:
            imm = sign * (action_values[m] - node_value)
            node.last_immediate_regret[m] = imm
            node.regret_sum[m] = max(0.0, node.regret_sum.get(m, 0.0) + imm)

        node.visits += 1
        return node_value

    else:
        # ---- Opponent's node: sample ONE move from their current strategy ----
        weights = [strategy.get(m, 0.0) for m in moves]
        total_w = sum(weights)
        if total_w == 0:
            sampled_move = random.choice(moves)
        else:
            sampled_move = random.choices(moves, weights=weights, k=1)[0]

        if sampled_move not in node.children:
            new_board, _ = make_move(node.board, sampled_move)
            child = CFRNode(new_board, 1 - node.side, parent=node, move=sampled_move)
            node.children[sampled_move] = child

        child_value = _cfr_traverse(
            node.children[sampled_move], player_side, traverser, depth - 1)

        node.visits += 1
        return child_value


def cfr_in_world(root_board, player_side, num_iterations=10):
    """
    External Sampling MCCFR with PCFR+ updates in one determinized world.

    Alternating updates (even t → player_side is the traverser, odd t → opponent)
    reduce variance and accelerate practical convergence.

    Returns {move: probability} – the *last-iterate* strategy at the root for
    player_side's moves.  PCFR+ last iterates converge to Nash equilibrium
    (no time-averaging required).
    """
    root = CFRNode(root_board, player_side)
    opp_side = 1 - player_side

    for t in range(num_iterations):
        traverser = player_side if (t % 2 == 0) else opp_side
        _cfr_traverse(root, player_side, traverser, CFR_TREE_DEPTH)

    # Last iterate: return the current strategy (not the time-average)
    return root.get_strategy()


def belief_state_cfr(belief_state, observed_board, player_side, time_limit=5.0):
    """
    IS-PCFR+ (Information-Set Predictive CFR+) over the full belief state.

    For each world sampled from the belief state, runs External Sampling MCCFR
    with PCFR+ updates (PRM+ clamping + predictive strategy) to compute the
    last-iterate strategy.  Move probabilities are aggregated across all worlds
    and the move with the highest combined probability is returned.

    Key differences from IS-CFR:
      - PRM+: cumulative regrets are clamped to ≥ 0 (no negative accumulation).
      - Predictive strategy: each iteration's strategy is informed by the
        immediate regret from the previous iteration (faster convergence).
      - Last-iterate play: the current strategy is used directly, with no
        time-averaging, giving stronger practical performance.
    """
    start_time = time.time()

    candidate_moves = generate_moves(observed_board, player_side)
    if not candidate_moves:
        return None

    worlds = belief_state.sample() if belief_state.size() > 0 else [observed_board]
    n_worlds = len(worlds)

    # Each CFR iteration at depth=1 explores ~len(moves) children at the root,
    # so the effective simulation count per iteration scales with the branching
    # factor.  We scale down iterations_per_world accordingly to keep total
    # rollout count comparable to the former IS-GRAVE budget.
    estimated_branching_factor = max(1, len(candidate_moves))
    raw_sims = max(BELIEF_MIN_SIMS_PER_WORLD,
                   min(BELIEF_MAX_SIMS_PER_WORLD,
                       BELIEF_TOTAL_SIM_BUDGET // max(1, n_worlds)))
    # Minimum of 2 so that both players get at least one alternating update
    # each (even iteration → player_side traverses, odd → opponent traverses).
    iterations_per_world = max(2, raw_sims // estimated_branching_factor)

    global_strategy = {m: 0.0 for m in candidate_moves}
    worlds_processed = 0

    for world in worlds:
        if time.time() - start_time > time_limit:
            break
        avg_strategy = cfr_in_world(world, player_side,
                                    num_iterations=iterations_per_world)
        for move, prob in avg_strategy.items():
            if move in global_strategy:
                global_strategy[move] += prob
        worlds_processed += 1

    total_weight = sum(global_strategy.values())
    if total_weight == 0:
        return candidate_moves[0]

    best_move = max(global_strategy, key=global_strategy.get)
    best_score = global_strategy[best_move] / max(1, worlds_processed)
    elapsed = time.time() - start_time
    total_iters = worlds_processed * iterations_per_world

    print(f"# IS-PCFR+: {worlds_processed} worlds, {total_iters} iterations, "
          f"{elapsed:.2f}s", flush=True)
    print(f"# Best move {sq_to_alg(best_move[0])}{sq_to_alg(best_move[1])}, "
          f"est strategy prob {best_score:.3f}", flush=True)

    return best_move


# Draw the board in a human-readable format
def draw_board(board):
    """Draw the board from white's perspective (rank 8 at top)"""
    lines = []
    lines.append("  +---+---+---+---+---+---+---+---+")
    for rank in range(7, -1, -1):  # 8 to 1
        row = f"{rank+1} |"
        for file in range(8):  # a to h
            sq = rank * 16 + file
            piece = board[sq]
            if piece == FOG:
                char = 'f'
            elif piece == 0:
                char = ' '
            elif piece > 0:  # white
                char = PIECE_CHARS[piece]
            else:  # black
                char = PIECE_CHARS[-piece].lower()
            row += f" {char} |"
        lines.append(row)
        lines.append("  +---+---+---+---+---+---+---+---+")
    lines.append("    a   b   c   d   e   f   g   h")
    return "\n".join(lines)


# Main engine loop (XBoard protocol)
def main():
    print("feature ping=1 myname=\"FoWMT v0.4 (IS-PCFR+)\" done=1", flush=True)
    observed_board = None
    player_side = None
    side_to_move = WHITE  # track which side is to move
    force_mode = False
    time_per_move = 40.0  # default time per move in seconds
    white_in_hand = {PAWN: 0, KNIGHT: 0, BISHOP: 0, ROOK: 0, QUEEN: 0, KING: 0}
    black_in_hand = {PAWN: 0, KNIGHT: 0, BISHOP: 0, ROOK: 0, QUEEN: 0, KING: 0}

    # Belief state – initialised lazily once we know which side we play
    belief = None


    last_fen = None   # most recently received setboard FEN string

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
            initial_fen = "ffffffff/ffffffff/ffffffff/ffffffff/8/8/PPPPPPPP/RNBQKBNR[] w - - 0 1"
            # To make initial fog, replace opponent ranks with f
            if player_side is None:
                # assume we play white first? but wait for color
                pass
        elif cmd[0] == "setboard":
            fen = " ".join(cmd[1:])
            last_fen = fen
            # Parse once; if a belief state already exists update it from the
            # same FEN so we avoid redundant parsing and keep history intact.
            if belief is not None:
                observed_board, white_in_hand, black_in_hand = \
                    belief.update_observation(fen)
                # side_to_move comes from a fresh parse (belief doesn't expose it)
                _, side_to_move, *_ = parse_extended_fen(fen, player_side)
            else:
                parsed = parse_extended_fen(fen, player_side)
                observed_board = parsed[0]
                side_to_move = parsed[1]
                white_in_hand = parsed[6]
                black_in_hand = parsed[7]
            print("Hint: position set", flush=True)
        elif cmd[0] == "force":
            force_mode = True
        elif cmd[0] == "go":
            force_mode = False
            player_side = side_to_move  # current to move is us

            # Ensure we have a belief state for this side; if player_side changed
            # (e.g. engine switched from white to black), recreate it and seed it
            # from the last known FEN so observation history is properly recorded.
            if belief is None or belief.player_side != player_side:
                belief = BeliefState(player_side, max_worlds=100)
                if last_fen is not None:
                    observed_board, white_in_hand, black_in_hand = \
                        belief.update_observation(last_fen)
                elif observed_board is not None:
                    belief.worlds = enumerate_worlds(
                        observed_board, player_side,
                        white_in_hand, black_in_hand, 100)

            best = belief_state_cfr(
                belief, observed_board, player_side, time_limit=time_per_move)
            if best:
                prom_char = ""
                if best[2]:
                    prom_char = PIECE_CHARS[best[2]].lower()
                move_str = sq_to_alg(best[0]) + sq_to_alg(best[1]) + prom_char
                print("move " + move_str, flush=True)
                # Propagate our move through the belief state
                belief.propagate_our_move(best)
                # Update the observed board locally too
                observed_board, _ = make_move(observed_board, best)
                # Update side to move after our move
                side_to_move = 1 - side_to_move
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
                # Propagate opponent's move through belief state
                if belief is not None:
                    belief.propagate_opponent_move(move)
                # Update side to move after opponent's move
                side_to_move = 1 - side_to_move
                # Update observations? For simplified, we don't update visibility here, assume user provides updated fen after opponent
        elif cmd[0] == "ping":
            print("pong " + cmd[1], flush=True)
        elif cmd[0] == "d":
            # Draw the board
            if observed_board:
                print(draw_board(observed_board), flush=True)
            else:
                print("# No board set. Use 'setboard' first.", flush=True)
        elif cmd[0] == "st":
            # Set time per move in seconds
            if len(cmd) > 1:
                try:
                    time_per_move = float(cmd[1])
                    print(f"# Time per move set to {time_per_move} seconds", flush=True)
                except ValueError:
                    print("# Error: st requires a numeric argument", flush=True)
            else:
                print("# Error: st requires time in seconds", flush=True)
        elif cmd[0] == "help":
            # Display available commands
            help_text = """# Available commands:
# xboard         - Enable XBoard mode
# protover N     - Set protocol version
# new            - Start new game
# setboard FEN   - Set position from extended FEN (with 'f' for fog squares)
# force          - Enter force mode (do not think)
# go             - Start thinking and make a move
# usermove MOVE  - Receive opponent's move (e.g., usermove e2e4)
# d              - Draw the current board
# st TIME        - Set time per move in seconds (default: 5.0)
# ping N         - Ping (responds with pong N)
# help           - Show this help message
# quit           - Exit the engine"""
            print(help_text, flush=True)
        elif cmd[0] == "quit":
            break


if __name__ == "__main__":
    main()