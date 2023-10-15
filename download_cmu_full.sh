#!/bin/bash

mkdir -p /data

wget -O /data/processed.tar.gz URL
tar -xzf /data/processed.tar.gz -C /data

echo "Done."