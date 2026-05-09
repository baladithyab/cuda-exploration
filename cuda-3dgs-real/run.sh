#!/bin/bash
set -e
cd "$(dirname "$0")"
./rasterize ../oxide-3dgs-real/scenes/utsuho_plush.ply 2>&1 | tee run.log
