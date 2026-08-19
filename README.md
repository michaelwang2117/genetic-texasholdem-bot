[README.md](https://github.com/user-attachments/files/31235188/README.md)
# Poker Evolver Prototype

## Prerequisites
- Python 3.9+
- Install Python deps:
  pip install -r requirements.txt
- Install OpenSpiel (pyspiel) per OpenSpiel docs so `import pyspiel` works. See OpenSpiel repository for build instructions. 

## Run server (check the port 43535 is on your machine free to use first)
python server.py

## Run GA evolver (in separate terminal)
python ga_evolver.py

## last step
Open http://localhost:43535

## Remark
- This is a coding project inspired by poker game I played during my universities years in economics/B&F, stats and data science. This coding project has to be refined on frontend, i.e. not complete on frontend of game rendering during pytorch training sessions - training sessions in terminal are working normally.
  
- The idea is based on genetic evolutionary progress - the selector choose by each round of training the winner bot, let this bot reproduce, then selective process applied again, an iterative process of winner of previous generations. In machine learning is call genetic evolutionary algorithm.
