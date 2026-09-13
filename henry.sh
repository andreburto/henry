#!/usr/bin/env sh

set -eu

script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$script_directory"

docker build --tag henry-local-validate .
docker run --rm -it \
  -v "$PWD/.env:/app/.env:ro" \
  henry-local-validate