#!/bin/bash

particles=(500 1000 2500 5000)
for num in "${particles[@]}"; do 
    for i in $(ls -d ../data/log/*); do
        #echo "python main.py $i $num &"
        python main.py $i $num &
    done
    wait
done
