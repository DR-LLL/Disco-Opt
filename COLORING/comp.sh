#!/bin/bash
set -e

g++ -O2 -std=c++17 -o solver solver.cpp
./solver