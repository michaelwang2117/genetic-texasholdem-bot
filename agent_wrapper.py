# agent_wrapper.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# --- Action abstraction mapping ---
# You can tune thresholds or mapping rules to match your pyspiel action ids.
ABSTRACT_ACTIONS = ["fold", "call", "small_raise", "big_raise"]
NUM_ABSTRACT = len(ABSTRACT_ACTIONS)

def map_concrete_to_abstract(state, action_id):
    # Heuristic mapping from concrete action id to abstract action.
    # This depends on the pyspiel game's action encoding.
    # For a robust system, replace with game-specific mapping.

    # Placeholder heuristic: use action_id numeric ranges
    if action_id == 0:
        return "fold"
    if action_id == 1:
        return "call"

    # raises: treat larger ids as bigger raises
    if action_id <= 10:
        return "small_raise"
    return "big_raise"

def choose_concrete_for_abstract(state, abstract_action):
    
    # Given a state and an abstract action, choose a legal concrete action
    # that best matches the abstract intent.
    # Strategy:
    #  - Inspect legal actions and their numeric ids
    #  - Prefer call/fold ids if present; for raises choose smallest or largest raise id
    
    legal = state.legal_actions()
    if not legal:
        return None

    # deterministic ordering
    legal_sorted = sorted(legal)
    if abstract_action == "fold":
        # prefer explicit fold id if present, else smallest id
        for a in legal_sorted:
            if map_concrete_to_abstract(state, a) == "fold":
                return a
        return legal_sorted[0]

    if abstract_action == "call":
        for a in legal_sorted:
            if map_concrete_to_abstract(state, a) == "call":
                return a
        # fallback: choose smallest non-fold
        for a in legal_sorted:
            if map_concrete_to_abstract(state, a) != "fold":
                return a
        return legal_sorted[0]

    if abstract_action == "small_raise":

        # choose the smallest raise id (heuristic)
        raise_ids = [a for a in legal_sorted if map_concrete_to_abstract(state, a).endswith("raise")]
        if not raise_ids:
            # fallback to call
            return choose_concrete_for_abstract(state, "call")
        return raise_ids[0]

    if abstract_action == "big_raise":
        raise_ids = [a for a in legal_sorted if map_concrete_to_abstract(state, a).endswith("raise")]
        if not raise_ids:
            return choose_concrete_for_abstract(state, "call")

        return raise_ids[-1]

    return legal_sorted[0]

# --- Encoder and policy ---
class PyspielStateEncoder:
    def __init__(self, obs_dim=128, max_action_space=128):
        self.obs_dim = obs_dim
        self.max_action_space = max_action_space

    def _pad_or_truncate(self, vec, length):
        vec = np.asarray(vec, dtype=np.float32)
        if vec.size >= length:
            return vec[:length]

        out = np.zeros(length, dtype=np.float32)
        out[:vec.size] = vec
        return out

    def encode(self, state, player):
        obs_parts = []
        try:
            if hasattr(state, "observation_tensor"):
                raw = state.observation_tensor(player)
                obs_parts.append(np.asarray(raw, dtype=np.float32).ravel())
            elif hasattr(state, "information_state_tensor"):
                raw = state.information_state_tensor(player)
                obs_parts.append(np.asarray(raw, dtype=np.float32).ravel())
        except Exception:
            pass

        if not obs_parts:
            try:
                hist = state.history()
            except Exception:
                hist = str(id(state))

            seed = abs(hash(hist)) % (2**31 - 1)
            rng = np.random.RandomState(seed)
            fallback = rng.randn(max(32, self.obs_dim // 4)).astype(np.float32)
            obs_parts.append(fallback)
            numeric = []

            try:
                if hasattr(state, "pot"):
                    numeric.append(float(state.pot()))
            except Exception:
                pass

            try:
                if hasattr(state, "player_stacks"):
                    stacks = state.player_stacks()
                    numeric.extend([float(s) for s in stacks])
            except Exception:
                pass

            if numeric:
                obs_parts.append(np.asarray(numeric, dtype=np.float32))

        # legal-action mask
        legal = []
        try:
            legal = state.legal_actions()
        except Exception:
            try:
                if hasattr(state, "legal_actions_mask"):
                    mask = state.legal_actions_mask()
                    legal = [i for i, v in enumerate(mask) if v]
            except Exception:
                legal = []

        max_action_id = 0
        if legal:
            max_action_id = max(legal)
        mask_len = max(self.max_action_space, max_action_id + 1, 1)
        mask = np.zeros(mask_len, dtype=np.float32)

        for a in legal:
            if 0 <= a < mask_len:
                mask[a] = 1.0

        combined = np.concatenate([p.ravel() for p in obs_parts] + [mask.ravel()])
        final = self._pad_or_truncate(combined, self.obs_dim)
        return final, mask  # mask returned for mapping

class PositionAwareMLP(nn.Module):
    
    # PyTorch MLP that conditions on seat position via an embedding.
    # Outputs logits over NUM_ABSTRACT abstract actions.
    
    def __init__(self, input_dim=128, hidden_dim=128, seat_embed_dim=16, num_seats=9, num_abstract=NUM_ABSTRACT):
        super().__init__()
        self.seat_embed = nn.Embedding(num_seats, seat_embed_dim)
        self.fc1 = nn.Linear(input_dim + seat_embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_abstract)

    def forward(self, obs_tensor, seat_idx):
        # obs_tensor: torch tensor shape (input_dim,)
        seat_emb = self.seat_embed(torch.tensor(seat_idx, dtype=torch.long))
        x = torch.cat([obs_tensor, seat_emb], dim=0)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        logits = self.head(x)
        return logits

class GenomeAgent:
    
    # Genome encodes PyTorch model parameters (flattened). We reconstruct model weights from genome.
    # For simplicity we initialize a model and load genome into its parameters by reshaping.
    
    def __init__(self, genome, seat=0, obs_dim=128, hidden_dim=128, seat_embed_dim=16, num_seats=9):
        self.genome = np.asarray(genome, dtype=np.float32)
        self.seat = seat
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.seat_embed_dim = seat_embed_dim
        self.num_seats = num_seats
        self.encoder = PyspielStateEncoder(obs_dim=obs_dim, max_action_space=128)

        # build model and load genome
        self.model = PositionAwareMLP(input_dim=obs_dim, hidden_dim=hidden_dim, seat_embed_dim=seat_embed_dim, num_seats=num_seats)
        self._load_genome_to_model(self.genome)

    def _load_genome_to_model(self, genome):
        
        # Map genome floats into model parameters deterministically.
        # If genome is too short, remaining params are left as initialized.
        # If genome is longer, extra values are ignored.
        
        ptr = 0
        for p in self.model.parameters():
            numel = p.numel()
            if ptr + numel <= genome.size:
                vals = genome[ptr:ptr + numel]
                arr = vals.reshape(p.shape)
                p.data.copy_(torch.from_numpy(arr))
                ptr += numel
            else:
                # partial fill if possible
                remaining = genome.size - ptr
                if remaining > 0:
                    vals = genome[ptr:ptr + remaining]
                    flat = p.data.view(-1)
                    flat[:remaining] = torch.from_numpy(vals)
                    ptr += remaining
                break

    def action(self, state):
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        obs_np, mask = self.encoder.encode(state, state.current_player())
        obs_t = torch.from_numpy(obs_np.astype(np.float32))
        logits = self.model(obs_t, state.current_player()).detach().numpy()

        # masked softmax over abstract actions
        # compute abstract action probabilities
        exps = np.exp(logits - np.max(logits))
        probs = exps / (exps.sum() + 1e-12)

        # sample abstract action
        abstract_idx = np.random.choice(len(probs), p=probs)
        abstract_action = ABSTRACT_ACTIONS[abstract_idx]

        # map to concrete action
        concrete = choose_concrete_for_abstract(state, abstract_action)
        if concrete is None:
            
            # fallback to random legal
            return random.choice(legal)
        return concrete
