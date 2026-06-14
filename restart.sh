#!/usr/bin/env bash
# Restart the local Jekyll preview (needed after editing _config.yml,
# layouts, includes, or SCSS — Jekyll does not hot-reload these).
set -e
cd "$(dirname "$0")"
docker compose down
docker compose up
