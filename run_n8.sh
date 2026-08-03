#!/bin/bash
cd /mnt/c/Users/chari/Documents/Projects/tc-fhe
export LD_LIBRARY_PATH=/home/charis/openfhe-install/lib
./build-bgv-batched/examples/bgv_batched_closure --graph=data/graph_N8.txt
echo "REAL_EXIT_CODE=$?"
