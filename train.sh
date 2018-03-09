#!/bin/bash
python main.py --dataset esc50 --netType envnetv2 --BC --strongAugment --save results_esc50v2_7 --GAP --l1reg 1e-7 --data /media/moreaux/Data/Dataset/ENVNET_DB/
python main.py --dataset esc50 --netType envnetv2 --BC --strongAugment --save results_esc50v2_6 --GAP --l1reg 1e-6 --data /media/moreaux/Data/Dataset/ENVNET_DB/
python main.py --dataset esc50 --netType envnetv2 --BC --strongAugment --save results_esc50v2_5 --GAP --l1reg 1e-5 --data /media/moreaux/Data/Dataset/ENVNET_DB/
