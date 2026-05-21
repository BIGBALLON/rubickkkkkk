# Fused-query slow tests (`test_fused_query_e2e.py`)

Optional local fixture — omitted from git (too large). To run:

```bash
RUBICK_RUN_SLOW=1 pytest tests/test_fused_query_e2e.py
```

```text
tests/fixtures/fused_e2e/
  notes/          # a few .md / .txt files
  images/
    cat-portrait.jpg
    mars-hubble.jpg
    jupiter-juno.jpg
    earth-blue-marble.jpg   # optional
```

Override path: `export RUBICK_FUSED_FIXTURE_DIR=/path/to/fixture`

Without this tree the module skips cleanly.
