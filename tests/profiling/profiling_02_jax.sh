#!/usr/bin/env sh
# Basic profiling for mat_dnf

py-spy record -o profile_02_jax.speedscope.json -n -i -f speedscope -- python profiling_02_jax.py
py-spy record -o profile_02_jax.svg -n -i -- python profiling_02_jax.py