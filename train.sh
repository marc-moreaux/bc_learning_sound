#!/bin/bash
for ITER in 1 2 3 
do

python3 main.py --dataset kitchen20 --netType envnetv2 --data /media/moreaux-gpu/Data/Dataset/ESC/ --save /media/moreaux-gpu/Data/envnet_results/kitchen20_ev2/$ITER
python3 main.py --dataset kitchen20 --netType envnetv2 --BC --strongAugment --data /media/moreaux-gpu/Data/Dataset/ESC/ --save /media/moreaux-gpu/Data/envnet_results/kitchen20_ev2_strong_BC/$ITER
python3 main.py --dataset kitchen20 --netType envnet --data /media/moreaux-gpu/Data/Dataset/ESC/ --save /media/moreaux-gpu/Data/envnet_results/kitchen20_ev/$ITER
python3 main.py --dataset kitchen20 --netType envnet --BC --strongAugment --data /media/moreaux-gpu/Data/Dataset/ESC/ --save /media/moreaux-gpu/Data/envnet_results/kitchen20_ev_strong_BC/$ITER

done

