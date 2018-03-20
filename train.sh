#!/bin/bash
# python main.py --dataset esc10 --netType envnet --BC --strongAugment --save results/DB2_esc10_6c --GAP --l1reg 1e-6 --data /media/moreaux/Data/Dataset/ENVNET_DB/
python main.py --dataset esc10 --netType envnet --BC --strongAugment --save results/DB2_esc10_7c --GAP --l1reg 1e-7 --data /media/moreaux/Data/Dataset/ENVNET_DB/
python main.py --dataset esc10 --netType envnet --BC --strongAugment --save results/DB2_esc10_8c --GAP --l1reg 1e-8 --data /media/moreaux/Data/Dataset/ENVNET_DB/
