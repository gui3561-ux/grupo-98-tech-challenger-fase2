#!/bin/bash

set -e

mkdir -p data/raw

curl -L -o data/raw/ml-100k.zip \
  https://files.grouplens.org/datasets/movielens/ml-100k.zip

unzip -o data/raw/ml-100k.zip -d data/raw
