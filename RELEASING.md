# Releasing ilp4lid

Test on TestPyPI first, then publish for real. A version number can only ever be uploaded
**once** to an index — there is no overwrite and no re-use after deletion — so the plan below
spends release candidates on TestPyPI and keeps `0.1.0` pristine for PyPI.

## 0. One-time setup

TestPyPI and PyPI are **separate** sites with separate accounts and tokens.

1. Register at <https://test.pypi.org/account/register/> and <https://pypi.org/account/register/>.
2. Enable 2FA on both (required to upload).
3. Create an API token on each: Account settings → API tokens → *Add API token*
   (scope "Entire account" for a first upload; re-scope to the project afterwards).
4. Keep them separate — a PyPI token is rejected by TestPyPI and vice versa:

```bash
export UV_PUBLISH_TOKEN_TEST="pypi-…"   # from test.pypi.org
export UV_PUBLISH_TOKEN_REAL="pypi-…"   # from pypi.org
```

## 1. Pre-flight

```bash
cd ~/ILP4LID
uv sync --extra mpi                       # dev group brings pytest
source .venv/bin/activate                 # REQUIRED: puts pytest and the ilp4lid-* commands on PATH
pytest tests -q                           # must be all green
ilp4lid-demo "Ich habe gestern ein meeting gehabt, it was really boring."
ilp4lid-batch --input dat/processed/DEV_1000.csv --out /tmp/preds.csv --limit 20   # batch run
```

Stay in this activated shell for the rest of section 1. Without `activate`, `ilp4lid-demo` fails
with `command not found`: the console scripts live in `.venv/bin`, which is only on `PATH` while
the venv is active.

Confirm the version is what you intend, in **both** places:

```bash
grep '^version' pyproject.toml
grep __version__ ilp4lid/__init__.py
```

Build clean — `dist/` is uploaded wholesale, so stale files from an earlier build would go too:

```bash
rm -rf dist && uv build
uvx twine check dist/*                    # expect PASSED twice
```

Check the wheel contains the package and nothing else:

```bash
python -c "import zipfile,glob;print(*sorted(zipfile.ZipFile(glob.glob('dist/*.whl')[0]).namelist()),sep='\n')"
```

Expect `ilp4lid/*.py`, `dist-info/` (with `licenses/LICENSE` and `entry_points.txt`) — and no
`analysis/`, `dat/`, `tests/` or `scripts/`.

## 2. Upload to TestPyPI

Use a release-candidate version so `0.1.0` stays free on PyPI. Bump both files to `0.1.0rc1`,
then:

```bash
rm -rf dist && uv build
uv publish --publish-url https://test.pypi.org/legacy/ --token "$UV_PUBLISH_TOKEN_TEST" --dry-run
uv publish --publish-url https://test.pypi.org/legacy/ --token "$UV_PUBLISH_TOKEN_TEST"
```

`--dry-run` validates the files without uploading. Without a token uv first tries trusted
publishing and errors out — pass `--token` explicitly.

## 3. Verify the TestPyPI install

In a throwaway environment, never the project venv. The dependencies
(`fasttext-wheel`, `pyomo`, …) are **not** on TestPyPI, so point the extra index at real PyPI:

```bash
cd /tmp && rm -rf rc && uv venv rc --python 3.12 && source rc/bin/activate
uv pip install --index-url https://test.pypi.org/simple/ \
               --extra-index-url https://pypi.org/simple/ ilp4lid==0.1.0rc1
python -c "import ilp4lid; print(ilp4lid.__version__, sorted(ilp4lid.CONFIGS))"
ilp4lid-demo "hola mundo, this is a test"
deactivate
```

That exercises what a user actually gets: the metadata resolves, the console scripts are on PATH,
and the demo runs. Iterate with `rc2`, `rc3`, … if anything is wrong.

## 4. Release to PyPI

```bash
# set version back to 0.1.0 in pyproject.toml AND ilp4lid/__init__.py
rm -rf dist && uv build
uvx twine check dist/*
git add -A && git commit -m "release: v0.1.0" && git push origin main
git tag -a v0.1.0 -m "ilp4lid v0.1.0" && git push origin v0.1.0
uv publish --token "$UV_PUBLISH_TOKEN_REAL"
```

Tag before publishing: the tag is what `pip install git+…@v0.1.0` and the HF Space resolve
against, and it makes the published artifact reproducible.

Then confirm the real thing:

```bash
cd /tmp && rm -rf rel && uv venv rel --python 3.12 && source rel/bin/activate
uv pip install ilp4lid && python -c "import ilp4lid; print(ilp4lid.__version__)"
deactivate
```

## 5. Gotchas

- **Versions are immutable.** Yanking hides a release but never frees the number. Bump, never retry.
- **`rm -rf dist` every time.** `uv publish` defaults to `dist/*` and will happily upload leftovers.
- **Keep the two `__version__` values in sync** — `pyproject.toml` decides the filename,
  `ilp4lid/__init__.py` is what users see at runtime.
- **`requires-python = ">=3.12"`** — test in a 3.12+ venv or the install just resolves to nothing.
- **`numpy<2` is a hard cap** (fasttext 0.9.2 breaks under numpy 2). It will conflict for users on
  modern numpy; that is the most likely install complaint.
- **TestPyPI is periodically pruned** and is not a backup — treat it as scratch space.
