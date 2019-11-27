#!/bin/bash
for r_dir in /media/moreaux-gpu/Data/envnet_results/*/1
do
python3 ./test.py --save $r_dir
done
