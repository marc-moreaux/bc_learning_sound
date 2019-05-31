#!/bin/bash
#python main.py --dataset esc50 --netType envnetv2 --BC --strongAugment --data /media/moreaux-gpu/Data/Dataset/ENVNET_DB/ --inputTime 1.5 --save ./results/esc50_trash
#python main.py --dataset esc50 --netType envnetv2 --BC --strongAugment --data /media/moreaux-gpu/Data/Dataset/ENVNET_DB/ --GAP --l1reg 1e-6 --inputTime 1.5 --save ./results/esc50_gap_6
python main.py --dataset esc50 --netType envnetv2 --BC --strongAugment --data /media/moreaux-gpu/Data/Dataset/ENVNET_DB/ --GAP --bypass --l1reg 1e-6 --inputTime 1.5 --save ./results/esc50_gap_bp_6
