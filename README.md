[README.md](https://github.com/user-attachments/files/31235188/README.md)
# Poker Evolver Prototype

## Prerequisites
- Python 3.9+
- Install Python deps:
  pip install -r requirements.txt
- Install OpenSpiel (pyspiel) per OpenSpiel docs so `import pyspiel` works. See OpenSpiel repository for build instructions. 

## Run server (check the port 43535 is on your machine free to use first)
python server.py
Open http://localhost:43535



## Run GA evolver (in separate terminal)
python -c "from ga_evolver import ga_loop; ga_loop()"

The GA will write the best genome to best_genome.pkl. The server reads GA stats via socket events (you can extend the GA to emit socket messages to the server).

python -c "from ga_evolver import ga_loop; ga_loop()"

## Remark
This is a coding project inspired by poker game I played during my universities years in economics/B&F, stats and data science. This coding project has to be refined on frontend, i.e. not complete on frontend of game rendering during DNN training sessions - training sessions in terminal are working.
