#!/bin/bash
# After the v7 run: score a spread of checkpoints on the three dev splits (never a benchmark),
# then the full evaluation of the final checkpoint (v6's best on accuracy was its final step).
cd /mnt/c/dev/OpenJEV
until grep -qa "TRAIN EXIT" .v7_train.log; do sleep 60; done
grep -qa "TRAIN EXIT 0" .v7_train.log || { echo "CHAIN STOP: training failed" > .v7_chain.log; exit 1; }
echo "select start $(date +%H:%M)" > .v7_chain.log
bash .v7_select.sh best step_400 step_800 step_1200 step_1600 final
echo "final start $(date +%H:%M)" >> .v7_chain.log
bash .v7_final.sh
echo "CHAIN DONE $(date +%H:%M)" >> .v7_chain.log
