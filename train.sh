#!/bin/bash
python main.py --dataset esc10 --netType envnet --BC --strongAugment --data /media/moreaux/Data/Dataset/ENVNET_DB/ --inputTime 2.5 --save ./results/esc10_att
python main.py --dataset esc10 --netType envnet --BC --strongAugment --data /media/moreaux/Data/Dataset/ENVNET_DB/ --GAP --l1reg 1e-6 --inputTime 2.5 --save ./results/esc10_att_6
