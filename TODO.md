# TODO

## Documentation CI guardrail (deferred — do not enable GitHub CI yet)

Keep `doc/` honest automatically once we're ready to turn on GitHub Actions.
Two complementary checks:

### 2a. Strict docs build on PR
GitHub Actions job that runs the same strict build we already use locally:

```
sphinx-build -b html -W --keep-going doc doc/_build/html
```

Catches broken cross-references (`:doc:`, `:ref:`, MyST `#anchor` links), RST
heading-underline bugs, files missing from the toctree, and bad `{include}`
paths (e.g. if `REST_API.md` / `TASK_CONFIG.md` get renamed).

Suggested `.github/workflows/docs.yml` — mirror `.readthedocs.yaml`
(Python 3.11, `doc/requirements.txt`), trigger only on `paths: ['doc/**', '*.md']`,
so a green CI ≈ a green Read the Docs build. Does not need the Django app
importable (the included `.md` files are static).

### 2b. Test that docs match code enumerations
A pytest in the existing suite (no extra infra) asserting that enumerations
documented in `TASK_CONFIG.md` are a subset of the code constants
(`supported_catalogs`, `supported_templates`, `supported_filters` in
`stdweb/processing/constants.py`). This is what would have caught the stale
`apass` catalog and the `zogy` -> `sfft` rename. Make it lenient on *additions*
in code but strict on *invented/stale* doc entries. Open decision: how strict /
how to parse the doc (a small structured block the test reads is more robust
than substring matching).

### Notes
- Ordering: 2a first (~15 min, structural), then 2b (~30 min, factual).
- If we later autogenerate reference tables from `constants.py` (the "option 3"
  idea), the 2a job would additionally need `pip install -r requirements.txt`
  and `DJANGO_SETTINGS_MODULE` so the app is importable at build time.
- Context: `doc/` is published to Read the Docs and now `{include}`s the
  root `REST_API.md` / `TASK_CONFIG.md` as the single source of truth.
