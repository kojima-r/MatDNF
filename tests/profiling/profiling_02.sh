#!/usr/bin/env sh
# Basic profiling for mat_dnf

py-spy record -o profile_02.speedscope.json -n -i -f speedscope -- python profiling_02.py
py-spy record -o profile_02.svg -n -i -- python profiling_02.py