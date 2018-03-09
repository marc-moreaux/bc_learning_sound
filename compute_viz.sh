#!/bin/bash
for r_dir in results*
do
python ./test.py --save $r_dir
done
