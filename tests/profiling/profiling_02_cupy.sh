#!/usr/bin/env sh
# Basic profiling for mat_dnf

py-spy record -o profile_02_cupy.speedscope.json -n -i -f speedscope -- python profiling_02_cupy.py
py-spy record -o profile_02_cupy.svg -n -i -- python profiling_02_cupy.py