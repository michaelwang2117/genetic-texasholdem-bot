# server.py

import os
import time
import threading
import numpy as np
import random
import pickle
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit


# Try to import pyspiel; if not installed the server still runs for UI and GA stats.
try:
    import pyspiel
except Exception:
    pyspiel = None

app = Flask(__name__, static_folder='static', template_folder='templates')
socketio = SocketIO(app, async_mode='eventlet')

NUM_PLAYERS = 9
BEST_GENOME_PATH = "best_genome.pkl"
GA_STATS_PATH = "ga_stats.pkl"
HOF_PATH = "hall_of_fame.pkl"

table_state = {
    "players": [{"id": i, "stack": 1000, "seat": i, "avatar": f"/static/avatars/{i%6}.png"} for i in range(NUM_PLAYERS)],
    "pot": 0,
    "deck_count": 52,
    "dealer_pos": 0,
    "board_cards": [],
    "hole_cards": {i: [] for i in range(NUM_PLAYERS)},
    "phase": "waiting",
    "generation": 0,
    "best_fitness": 0.0
}

# -------------------------
# GA stats loader (existing)
# -------------------------
def load_stats():
    if os.path.exists(GA_STATS_PATH):
        try:
            with open(GA_STATS_PATH, "rb") as f:
                stats = pickle.load(f)
            table_state['generation'] = stats.get('generation', table_state['generation'])
            table_state['best_fitness'] = stats.get('best_fitness', table_state['best_fitness'])
        except Exception:
            pass

def stats_watcher():
    last_mtime = None
    while True:
        try:
            if os.path.exists(GA_STATS_PATH):
                mtime = os.path.getmtime(GA_STATS_PATH)
                if last_mtime is None or mtime != last_mtime:
                    load_stats()
                    socketio.emit('table_update', table_state, broadcast=True)
                    last_mtime = mtime
        except Exception:
            pass
        time.sleep(0.1)

watcher_thread = threading.Thread(target=stats_watcher, daemon=True)
watcher_thread.start()

# -------------------------
# Table broadcaster (existing)
# -------------------------
def broadcaster():
    while True:
        socketio.emit('table_update', table_state)
        time.sleep(0.2)

b_thread = threading.Thread(target=broadcaster, daemon=True)
b_thread.start()

# -------------------------
# Hall of Fame utilities (new)
# -------------------------
def load_hof():
    """Load hall of fame list of genomes (or empty list)."""
    if os.path.exists(HOF_PATH):
        try:
            with open(HOF_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return []
    return []

def compute_hof_matrix():
    # Quick head-to-head matrix among HOF members using a small number of hands.
    # Returns a dict with labels and matrix (rows: i vs columns: j average chips per hand).
    # This function tries to call simulate_match_deterministic from ga_evolver if available.
    # If ga_evolver is not importable, it returns zeros.
    
    hof = load_hof()
    n = len(hof)
    if n == 0:
        return {"labels": [], "matrix": []}

    matrix = [[0.0]*n for _ in range(n)]

    # Try to import deterministic simulator from ga_evolver; fallback to zeros if unavailable
    simulate_fn = None
    try:
        # import inside function to avoid circular imports at module load time
        from ga_evolver import simulate_match_deterministic
        simulate_fn = simulate_match_deterministic
    except Exception:
        simulate_fn = None

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0.0
            else:
                if simulate_fn is None:
                    matrix[i][j] = 0.0
                else:
                    try:
                        # For speed, use a small number of hands and deterministic seed
                        opponents = [hof[j]] * 8  # fill seats 1..8 with the same HOF genome j
                        score = simulate_fn(hof[i], opponents, num_hands=100, seed_base=1000 + i*100 + j)
                        matrix[i][j] = float(score)
                    except Exception:
                        matrix[i][j] = 0.0

    labels = [f"HOF_{k}" for k in range(n)]
    return {"labels": labels, "matrix": matrix}

def hof_broadcaster():
    # Background thread: watch HOF file and emit 'hof_update' when it changes.
    # The UI listens for 'hof_update' and renders the HOF list and head-to-head matrix.

    last_mtime = None
    while True:
        try:
            if os.path.exists(HOF_PATH):
                mtime = os.path.getmtime(HOF_PATH)
                if last_mtime is None or mtime != last_mtime:
                    hof_summary = compute_hof_matrix()
                    socketio.emit('hof_update', hof_summary, broadcast=True)
                    last_mtime = mtime
        except Exception:
            pass
        time.sleep(1.0)

hof_thread = threading.Thread(target=hof_broadcaster, daemon=True)
hof_thread.start()

# -------------------------------------------
# Flask routes and socket handlers (existing)
# -------------------------------------------
@app.route('/')
def index():
    return render_template('table.html')

@app.route('/static/avatars/<path:filename>')
def avatars(filename):
    return send_from_directory('static/avatars', filename)

@socketio.on('connect')
def on_connect():
    # send initial table and HOF info
    emit('table_update', table_state)
    
    # also send HOF summary immediately
    emit('hof_update', compute_hof_matrix())

@socketio.on('deal')
def on_deal(data):
    emit('table_update', table_state)

if __name__ == '__main__':
    print("Starting server on http://127.0.0.1:43535")
    socketio.run(app, host='127.0.0.1', port=43535, debug=False)
