# data/

`cache/` holds daily closes downloaded from Yahoo Finance on first run, one CSV
per ticker. It is not tracked: Yahoo's terms do not allow redistribution, so the
files are fetched on your machine rather than shipped with the repository.

Populate it by running any study — `python studies/retail_recipes.py` — which
downloads what it needs once and reads from disk afterwards. Every study then
re-runs offline.

The test suite never reads this directory. It runs on synthetic data with known
properties and never touches the network.
