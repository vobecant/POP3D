#!/bin/bash
wget -P ./data https://data.ciirc.cvut.cz/public/projects/2023POP3D/retrieval_benchmark.tar.gz
mkdir -p ./data
cd ./data 
tar -xzf retrieval_benchmark.tar.gz
mv retrieval_benchmark_release retrieval_benchmark
rm retrieval_benchmark.tar.gz
cd ..
