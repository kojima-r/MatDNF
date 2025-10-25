#!/usr/bin/env sh
# Basic profiling for mat_dnf

py-spy record -o profile.speedscope.json -n -i -f speedscope -- python profiling.py
py-spy record -o profile.svg -n -i -- python profiling.py