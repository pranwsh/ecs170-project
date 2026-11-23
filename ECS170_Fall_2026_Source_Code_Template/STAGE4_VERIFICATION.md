# Stage 4 verification

Stage 4 focused test suite: **55 passed, 9 subtests passed** (CPU, Python 3.14, 2026-11-22). The project Nix development shell does not include pytest; the tests were run with the Nix pytest executable and the development environment's site-packages on `PYTHONPATH`:

```sh
nix develop -c bash -c 'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; export PYTHONPATH="/nix/store/ssksfwjn7iykzvq4hc34gdl3sydlahwy-python3-3.14.8-env/lib/python3.14/site-packages:$PYTHONPATH"; cd ECS170_Fall_2026_Source_Code_Template; /nix/store/1p2fja7y7dy76d7jbkrc51wrlssdz6zf-python3.14-pytest-9.1.1/bin/pytest -q tests/test_stage4*.py'
```

The `/nix/store` paths above identify the particular local environment and are not portable. Passing fixture tests does not establish model quality or substitute for full dataset experiments.
