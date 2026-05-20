#!/bin/bash
set -e

cd "$(dirname "$0")"

clang++ -O2 -std=c++17 -Wall -Wextra -pedantic solver.cpp -o solver

./solver 590