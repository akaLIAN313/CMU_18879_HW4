#!/bin/bash
python3 pointElec_sim.py 0.5 0 700 0 500 
python3 pointElec_sim.py 1 300 1800 300 900 
python3 pointElec_sim.py 2 1000 9000 1000 5000 
python3 pointElec_sim.py 4 6000 30000 6000 17000 
python3 pointElec_sim.py 8 10000 170000 10000 80000 
python3 pointElec_sim.py 16 100000 500000 100000 300000
python3 uniform_sim.py 0 300 0 300
python3 plot_summary_pointElec.py

