#!/usr/bin/env bash

# Updates examples on the example directory
for d in bob.example.*; do
  echo "Processing example project ${d}..."
  tar cfj ${d}.tar.bz2 ${d};
  mv -fv ${d}.tar.bz2 ../bob/extension/data/;
  echo "Example project ${d} is updated"
done
