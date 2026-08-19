
# ga_evolver.py
import numpy as np
import multiprocessing as mp
import random
import time
import pickle
import os
import threading
import math
import socketio
import torch
import torch.optim as optim
import pyspiel
from agent_wrapper import map_concrete_to_abstract, choose_concrete_for_abstract

try:
    # Try loading with no params to see defaults or to trigger helpful error text
    game = pyspiel.load_game("universal_poker")
    print("Loaded universal_poker with defaults.")
except Exception as e:
    print("Error loading universal_poker:", e)
    # The exception text usually lists available parameters; print it for reference
    print(str(e))



from agent_wrapper import GenomeAgent, PositionAwareMLP, ABSTRACT_ACTIONS, choose_concrete_for_abstract

# --- Model dims (must match agent_wrapper defaults) ---
OBS_DIM = 128
HIDDEN_DIM = 128
SEAT_EMBED_DIM = 16
NUM_SEATS = 9
NUM_ABSTRACT = len(ABSTRACT_ACTIONS)

# --- seed normalization helper ---
def norm_seed(x):
    
    # Convert x to a deterministic non-negative 32-bit Python int suitable for
    # random.seed, numpy.random.seed, and torch.manual_seed.
    # Accepts numpy scalars, bytes, strings, floats, arrays, etc.
    
    try:
        s = int(x)
    except Exception:
        # fallback to Python hash for arbitrary objects
        s = abs(hash(x))
    # mask to 31 bits to ensure positive int in range
    return int(s & 0x7FFFFFFF)


# Compute required genome size deterministically from model parameter counts
def compute_genome_size():
    model = PositionAwareMLP(input_dim=OBS_DIM, hidden_dim=HIDDEN_DIM, seat_embed_dim=SEAT_EMBED_DIM, num_seats=NUM_SEATS, num_abstract=NUM_ABSTRACT)
    total = 0
    for p in model.parameters():
        total += p.numel()
    return total

GENOME_SIZE = compute_genome_size()

# GA config
POP_SIZE = 40
ELITE_K = 4
HALL_OF_FAME_K = 8
TOURNAMENT_K = 3
MUTATION_PROB = 0.02
MUTATION_STD = 0.03
EVAL_HANDS_QUICK = 200
EVAL_HANDS_MED = 1000
EVAL_HANDS_FULL = 3000
NUM_WORKERS = max(1, mp.cpu_count() - 1)

# Fold, Call, Pot-raise, All-in
GAME_STR = "universal_poker"
BEST_GENOME_PATH = "best_genome.pkl"
GA_STATS_PATH = "ga_stats.pkl"
HOF_PATH = "hall_of_fame.pkl"

SERVER_SOCKETIO_URL = os.environ.get("GA_SERVER_SOCKETIO_URL", "http://localhost:43536")
sio = socketio.Client(reconnection=True, logger=False, engineio_logger=False)

def _connect_socketio():
    while True:
        try:
            if not sio.connected:
                sio.connect(SERVER_SOCKETIO_URL, wait=True, transports=["websocket"])
            return
        except Exception:
            time.sleep(1.0)

_connect_thread = threading.Thread(target=_connect_socketio, daemon=True)
_connect_thread.start()

@sio.event
def connect():
    print("[GA->Server] Connected to SocketIO server.")

@sio.event
def disconnect():
    print("[GA->Server] Disconnected from SocketIO server.")

# Deterministic RNG helper
def make_rng(seed):
    return np.random.RandomState(seed), random.Random(seed)

# Initialize population
def init_population():
    pop = []
    for _ in range(POP_SIZE):
        # sample genome from normal distribution
        g = np.random.normal(0, 0.1, GENOME_SIZE).astype(np.float32)
        pop.append(g)
    return pop

# Evaluate one genome deterministically using a fixed seed
def simulate_match_deterministic(genome, opponents_genomes, num_hands, seed_base):
    """
    Deterministic simulation: RNG seeds derived from seed_base and hand index.
    Returns average chips per hand for seat 0.
    """
    if pyspiel is None:
        # deterministic pseudo-random based on genome bytes and seed
        seed = int(abs(hash(genome.tobytes())) % (2**31 - 1)) ^ seed_base
        rng = np.random.RandomState(seed)
        return float(rng.randn() * 0.5)

    game = pyspiel.load_game(GAME_STR)
    my_agent = GenomeAgent(genome, seat=0, obs_dim=OBS_DIM, hidden_dim=HIDDEN_DIM, seat_embed_dim=SEAT_EMBED_DIM, num_seats=NUM_SEATS)
    opp_agents = [GenomeAgent(g, seat=i+1, obs_dim=OBS_DIM, hidden_dim=HIDDEN_DIM, seat_embed_dim=SEAT_EMBED_DIM, num_seats=NUM_SEATS) for i, g in enumerate(opponents_genomes)]
    total_return = 0.0

    for h in range(num_hands):
        # seed per hand for determinism
        hand_seed = seed_base ^ (h + 1)
        np.random.seed(hand_seed)
        random.seed(hand_seed)
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                acts = [o[0] for o in outcomes]
                probs = [o[1] for o in outcomes]
                # deterministic sampling using numpy choice with seed
                a = np.random.choice(acts, p=np.array(probs)/sum(probs))
                state.apply_action(int(a))
                continue
            cur = state.current_player()
            if cur == 0:
                a = my_agent.action(state)
            else:
                opp = opp_agents[(cur - 1) % len(opp_agents)]
                a = opp.action(state)
            legal = state.legal_actions()

            if a not in legal:
                a = int(np.random.choice(legal))
            state.apply_action(int(a))
        returns = state.returns()
        total_return += returns[0]
    return float(total_return / num_hands)

# Evaluate genome against mixed opponents (population + HOF) deterministically
def evaluate_genome(genome, population, hall_of_fame, num_hands=EVAL_HANDS_QUICK, rounds=3, seed_offset=0):
    opponents_pool = []

    # sample from population (exclude genome if present)
    for _ in range(min(6, max(1, len(population)//2))):
        opponents_pool.append(random.choice(population))

    # add hall of fame
    for _ in range(min(len(hall_of_fame), 3)):
        opponents_pool.append(random.choice(hall_of_fame))
    total_return = 0.0
    total_hands = 0

    for r in range(rounds):
        # sample 8 opponents deterministically using seed_offset and r
        seed = seed_offset ^ (r + 1)
        rng = np.random.RandomState(seed)
        opponents = [opponents_pool[rng.randint(0, len(opponents_pool))] for _ in range(8)]
        avg = simulate_match_deterministic(genome, opponents, num_hands, seed_base=seed)
        total_return += avg * num_hands
        total_hands += num_hands
    return float(total_return / total_hands) if total_hands > 0 else 0.0

# GA operators
def tournament_select(pop, fitnesses, k=TOURNAMENT_K):
    idxs = random.sample(range(len(pop)), k)
    best = max(idxs, key=lambda i: fitnesses[i])
    return pop[best]

def crossover(a, b):
    mask = np.random.rand(len(a)) < 0.5
    child = np.where(mask, a, b)
    return child

def mutate(genome):
    mask = np.random.rand(len(genome)) < MUTATION_PROB
    genome[mask] += np.random.normal(0, MUTATION_STD, mask.sum())
    return genome

def save_pickle_atomic(obj, path):
    try:
        with open(path + ".tmp", "wb") as f:
            pickle.dump(obj, f)
        os.replace(path + ".tmp", path)
    except Exception:
        pass

def broadcast_stats(generation, best_fitness, mean_fitness, top_std):
    stats = {"generation": generation, "best_fitness": float(best_fitness), "mean_fitness": float(mean_fitness), "top_std": float(top_std)}
    try:
        if sio.connected:
            sio.emit('ga_update', stats)
    except Exception:
        pass
    save_pickle_atomic(stats, GA_STATS_PATH)

# --- Fine-tuning (Adam + REINFORCE) ---
def fine_tune_genome(genome, population, hall_of_fame, lr=1e-3, steps=50, batch_hands=200, seed_base=12345):
    """
    Convert genome to a PyTorch model, run REINFORCE on rollouts against sampled opponents.
    Returns a new genome (numpy array) with updated parameters.
    """
    # Build model and load genome into model parameters
    model = PositionAwareMLP(input_dim=OBS_DIM, hidden_dim=HIDDEN_DIM, seat_embed_dim=SEAT_EMBED_DIM, num_seats=NUM_SEATS, num_abstract=NUM_ABSTRACT)
    # load genome into model
    ptr = 0
    g = torch.from_numpy(genome)
    for p in model.parameters():
        numel = p.numel()
        if ptr + numel <= g.numel():
            vals = g[ptr:ptr + numel].view(p.shape)
            p.data.copy_(vals)
            ptr += numel
        else:
            remaining = g.numel() - ptr

            if remaining > 0:
                vals = g[ptr:ptr + remaining]
                flat = p.data.view(-1)
                flat[:remaining] = vals
                ptr += remaining
            break

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    # sample opponents for fine-tuning (mix of pop and HOF)
    opponents_pool = []
    for _ in range(6):
        opponents_pool.append(random.choice(population))
    for _ in range(min(len(hall_of_fame), 3)):
        opponents_pool.append(random.choice(hall_of_fame))

    for step in range(steps):
        # sample opponents for this batch deterministically
        seed = seed_base ^ step
        rng = np.random.RandomState(seed)
        opponents = [opponents_pool[rng.randint(0, len(opponents_pool))] for _ in range(8)]
        # run batch_hands rollouts and collect (logprob * return) for REINFORCE
        logprob_list = []
        returns_list = []

        for h in range(batch_hands):
            hand_seed = seed_base ^ (step + 1) ^ (h + 1)
            seed_int = norm_seed(hand_seed)
            np.random.seed(seed_int)
            random.seed(seed_int)

            # simulate one hand and record logprobs for actions taken by seat 0
            if pyspiel is None:
                # synthetic reward
                reward = float(np.random.randn() * 0.5)
                returns_list.append(reward)
                logprob_list.append(0.0)
                continue

            game = pyspiel.load_game(GAME_STR)
            state = game.new_initial_state()
            # build opponent agents
            opp_agents = [GenomeAgent(g, seat=i+1, obs_dim=OBS_DIM, hidden_dim=HIDDEN_DIM, seat_embed_dim=SEAT_EMBED_DIM, num_seats=NUM_SEATS) for i, g in enumerate(opponents)]
            hand_logprobs = []

            while not state.is_terminal():

                if state.is_chance_node():
                    outcomes = state.chance_outcomes()
                    acts = [o[0] for o in outcomes]
                    probs = [o[1] for o in outcomes]
                    a = np.random.choice(acts, p=np.array(probs)/sum(probs))
                    state.apply_action(int(a))
                    continue
                cur = state.current_player()

                if cur == 0:
                    # compute logits from model for seat 0
                    encoder = model  # reuse model for encoding via agent_wrapper's encoder is not used here
                    # Use the same encoder as agent_wrapper to get obs and mask
                    from agent_wrapper import PyspielStateEncoder

                    enc = PyspielStateEncoder(obs_dim=OBS_DIM, max_action_space=128)
                    obs_np, mask = enc.encode(state, 0)
                    obs_t = torch.from_numpy(obs_np.astype(np.float32))
                    logits = model(obs_t, 0)

                    # masked softmax
                    masked_logits = logits.clone()

                    # mask abstract actions that have no corresponding concrete legal action
                    legal = state.legal_actions()

                    # compute which abstract actions are available
                    abstract_mask = torch.zeros(NUM_ABSTRACT, dtype=torch.float32)


                    for a in legal:
                        abs_name = map_concrete_to_abstract(state, a)
                        if abs_name in ABSTRACT_ACTIONS:
                            abstract_mask[ABSTRACT_ACTIONS.index(abs_name)] = 1.0

                    # if no abstract available, allow all
                    if abstract_mask.sum() == 0:
                        abstract_mask = torch.ones(NUM_ABSTRACT)

                    # set illegal logits to -inf
                    masked_logits = masked_logits * abstract_mask + (1.0 - abstract_mask) * (-1e9)
                    probs = torch.softmax(masked_logits, dim=0)
                    m = torch.distributions.Categorical(probs)
                    a_idx = m.sample()
                    logp = m.log_prob(a_idx)
                    hand_logprobs.append(logp)

                    # map to concrete
                    abstract_action = ABSTRACT_ACTIONS[int(a_idx.item())]
                    concrete = choose_concrete_for_abstract(state, abstract_action)
                    if concrete is None:
                        concrete = random.choice(legal)

                    state.apply_action(int(concrete))
                else:
                    opp = opp_agents[(cur - 1) % len(opp_agents)]
                    a = opp.action(state)
                    legal = state.legal_actions()

                    if a not in legal:
                        a = random.choice(legal)

                    state.apply_action(int(a))

            returns = state.returns()
            reward = returns[0]

            # accumulate
            if len(hand_logprobs) > 0:
                logprob_list.append(torch.stack(hand_logprobs).sum())
            else:
                logprob_list.append(torch.tensor(0.0))

            returns_list.append(float(reward))

        # compute policy gradient loss: -E[logpi * (R - baseline)]
        returns_tensor = torch.tensor(returns_list, dtype=torch.float32)
        baseline = returns_tensor.mean()
        loss = 0.0

        for lp, R in zip(logprob_list, returns_list):
            loss = loss - lp * (R - baseline)

        loss = loss / max(1, len(returns_list))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # After fine-tuning, extract model params back into genome
    new_genome = np.zeros(GENOME_SIZE, dtype=np.float32)
    ptr = 0
    for p in model.parameters():
        numel = p.numel()
        vals = p.data.cpu().numpy().reshape(-1)
        if ptr + numel <= GENOME_SIZE:
            new_genome[ptr:ptr + numel] = vals
            ptr += numel
        else:
            remaining = GENOME_SIZE - ptr

            if remaining > 0:
                new_genome[ptr:ptr + remaining] = vals[:remaining]
                ptr += remaining
            break

    return new_genome

# Broadcast helper
def save_pickle_atomic(obj, path):
    try:
        with open(path + ".tmp", "wb") as f:
            pickle.dump(obj, f)
        os.replace(path + ".tmp", path)
    except Exception:
        pass

def broadcast_stats(generation, best_fitness, mean_fitness, top_std):
    stats = {"generation": generation, "best_fitness": float(best_fitness), "mean_fitness": float(mean_fitness), "top_std": float(top_std)}
    try:
        if sio.connected:
            sio.emit('ga_update', stats)
    except Exception:
        pass
    save_pickle_atomic(stats, GA_STATS_PATH)


# Main GA loop with HOF and fine-tuning
def ga_loop():
    pop = init_population()
    hof = []
    fitnesses = [0.0] * len(pop)
    generation = 0
    pool = mp.Pool(NUM_WORKERS)
    try:
        while True:
            generation += 1
            tasks = []
            for i, g in enumerate(pop):
                seed_offset = generation * 1000 + i
                tasks.append(pool.apply_async(evaluate_genome, (g, pop, hof, EVAL_HANDS_QUICK, 2, seed_offset)))

            for i, t in enumerate(tasks):
                try:
                    fitnesses[i] = t.get()
                except Exception:
                    fitnesses[i] = float(-1e6)

            mean_fitness = float(np.mean(fitnesses))
            ranked = sorted(range(len(pop)), key=lambda i: fitnesses[i], reverse=True)
            best_idx = ranked[0]
            best_genome = pop[best_idx]
            best_fitness = fitnesses[best_idx]
            top_std = float(np.std([fitnesses[i] for i in ranked[:max(1, ELITE_K)]]))
            print(f"[GA] Gen {generation} best {best_fitness:.4f} mean {mean_fitness:.4f} std_top {top_std:.4f}")

            # Save best genome
            save_pickle_atomic(best_genome, BEST_GENOME_PATH)

            # Update HOF: merge elites and keep top by quick eval
            elites = [pop[i].copy() for i in ranked[:ELITE_K]]
            combined = hof + elites
            if combined:
                comb_scores = []
                for idx, g in enumerate(combined):
                    comb_scores.append(evaluate_genome(g, pop, hof, EVAL_HANDS_QUICK, 2, generation * 100 + idx))
                order = sorted(range(len(combined)), key=lambda i: comb_scores[i], reverse=True)
                hof = [combined[i].copy() for i in order[:HALL_OF_FAME_K]]
            else:
                hof = elites[:HALL_OF_FAME_K]

            save_pickle_atomic(hof, HOF_PATH)

            # Broadcast stats
            broadcast_stats(generation, best_fitness, mean_fitness, top_std)

            # Fine-tune top genomes (small population) with Adam/REINFORCE
            top_to_finetune = [pop[i] for i in ranked[:max(2, ELITE_K)]]
            finetuned = []

            for idx, g in enumerate(top_to_finetune):
                print(f"[GA] Fine-tuning top candidate {idx+1}/{len(top_to_finetune)}")
                newg = fine_tune_genome(g, pop, hof, lr=1e-3, steps=20, batch_hands=100, seed_base=g.sum().astype(int) & 0x7fffffff)
                finetuned.append(newg)
                
            # Evaluate finetuned genomes and replace if improved
            for newg in finetuned:
                new_score = evaluate_genome(newg, pop, hof, EVAL_HANDS_MED, 3, generation * 1000)
                if new_score > best_fitness + 0.01:
                    print(f"[GA] Fine-tuned genome improved from {best_fitness:.4f} to {new_score:.4f}; replacing best.")
                    best_genome = newg
                    best_fitness = new_score
                    save_pickle_atomic(best_genome, BEST_GENOME_PATH)

            # Create next generation
            new_pop = [pop[i].copy() for i in ranked[:ELITE_K]]
            while len(new_pop) < POP_SIZE:
                p1 = tournament_select(pop, fitnesses)
                p2 = tournament_select(pop, fitnesses)
                child = crossover(p1, p2)
                child = mutate(child)
                new_pop.append(child)
            pop = new_pop

    finally:
        pool.close()
        pool.join()

if __name__ == "__main__":
    ga_loop()
