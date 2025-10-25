#!/usr/bin/env bash

for i in $(seq 0 4); do
  octave-cli --eval 'generate_test_arrays("../../tests/resources/mat_dnf_sam", '$i')'
done