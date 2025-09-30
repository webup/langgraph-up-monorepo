# Publishing Guide for langgraph-up-devkits

This guide covers how to publish `langgraph-up-devkits` to PyPI.

## Quick Start

### Prerequisites

1. **Install build tools** (one-time setup):
   ```bash
   cd libs/langgraph-up-devkits
   uv add --dev build twine
   ```

2. **Set up PyPI accounts**:
   - Test PyPI: https://test.pypi.org/account/register/
   - Production PyPI: https://pypi.org/account/register/

3. **Create API tokens**:
   - Log in to PyPI/TestPyPI
   - Go to Account Settings → API tokens
   - Create token with "Entire account" or project-specific scope
   - Save tokens securely

4. **Configure credentials** (choose one):

   **Option A: Using `.pypirc` (recommended)**
   ```bash
   cat > ~/.pypirc <<EOF
   [distutils]
   index-servers =
       pypi
       testpypi

   [pypi]
   username = __token__
   password = <your-pypi-token>

   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = __token__
   password = <your-test-pypi-token>
   EOF
   chmod 600 ~/.pypirc
   ```

   **Option B: Using environment variables**
   ```bash
   export TWINE_USERNAME=__token__
   export TWINE_PASSWORD=<your-token>
   ```

## Publishing Workflow

### Step 1: Pre-Publication Checklist

- [ ] Update version in `libs/langgraph-up-devkits/pyproject.toml`
- [ ] Update version in `libs/langgraph-up-devkits/src/langgraph_up_devkits/__init__.py`
- [ ] Update `README.md` with any new features
- [ ] Update GitHub URLs in `pyproject.toml` if needed
- [ ] Run all tests: `make test_libs`
- [ ] Run linting: `make lint_libs`
- [ ] Commit all changes: `git commit -am "chore: bump version to x.y.z"`
- [ ] Create git tag: `git tag vx.y.z`
- [ ] Push commits and tag: `git push && git push --tags`

### Step 2: Build Package

```bash
make build_devkits
```

This will:
- Clean old builds (`dist/`, `build/`, `*.egg-info`)
- Create new distribution packages in `libs/langgraph-up-devkits/dist/`
  - Source distribution (`.tar.gz`)
  - Wheel distribution (`.whl`)

### Step 3: Check Package

```bash
make check_devkits
```

This validates the package structure and metadata.

### Step 4: Test on Test PyPI (Recommended)

```bash
make publish_test_devkits
```

Then test installation:
```bash
pip install --index-url https://test.pypi.org/simple/ langgraph-up-devkits
```

**OR use the all-in-one command:**
```bash
make release_test_devkits
```

### Step 5: Publish to Production PyPI

```bash
make publish_devkits
```

This will:
1. Show a confirmation prompt (press Ctrl+C to cancel)
2. Upload to PyPI
3. Display installation command

**OR use the all-in-one command:**
```bash
make release_devkits
```

### Step 6: Verify Installation

```bash
pip install langgraph-up-devkits
python -c "from langgraph_up_devkits import load_chat_model; print('Success!')"
```

## Available Make Commands

| Command | Description |
|---------|-------------|
| `make build_devkits` | Build distribution packages |
| `make check_devkits` | Validate package with twine |
| `make publish_test_devkits` | Upload to Test PyPI |
| `make publish_devkits` | Upload to Production PyPI (with confirmation) |
| `make release_test_devkits` | Build and publish to Test PyPI |
| `make release_devkits` | Build and publish to Production PyPI |

## Manual Publishing (Alternative)

If you prefer to run commands manually:

```bash
# Navigate to package directory
cd libs/langgraph-up-devkits

# Clean and build
rm -rf dist/ build/ *.egg-info
uv run python -m build

# Check package
uv run python -m twine check dist/*

# Upload to Test PyPI
uv run python -m twine upload --repository testpypi dist/*

# Upload to Production PyPI
uv run python -m twine upload dist/*
```

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version (x.0.0): Breaking changes
- **MINOR** version (0.x.0): New features (backward compatible)
- **PATCH** version (0.0.x): Bug fixes (backward compatible)

Examples:
- `0.1.0` → `0.1.1` (bug fix)
- `0.1.1` → `0.2.0` (new feature)
- `0.2.0` → `1.0.0` (stable release or breaking change)

## Troubleshooting

### Issue: "File already exists" error on PyPI

**Solution**: You cannot re-upload the same version. Increment the version number in `pyproject.toml` and rebuild.

### Issue: Missing dependencies during build

**Solution**: Install build dependencies:
```bash
cd libs/langgraph-up-devkits
uv sync --all-extras
uv add --dev build twine
```

### Issue: Authentication failed

**Solution**:
1. Verify API token is correct
2. Check `.pypirc` permissions: `chmod 600 ~/.pypirc`
3. Ensure using `__token__` as username (not your account name)

### Issue: Package validation errors

**Solution**: Run `make check_devkits` to see detailed errors. Common issues:
- Missing `README.md`
- Invalid `pyproject.toml` format
- Missing required metadata fields

## GitHub Actions (Optional)

For automated publishing on release, create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install uv
      run: pip install uv

    - name: Build package
      working-directory: libs/langgraph-up-devkits
      run: |
        uv sync
        uv run python -m build

    - name: Publish to PyPI
      working-directory: libs/langgraph-up-devkits
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: uv run python -m twine upload dist/*
```

Then add your PyPI token to GitHub Secrets:
1. Go to repository Settings → Secrets → Actions
2. Add new secret: `PYPI_API_TOKEN` with your PyPI token

## Resources

- PyPI: https://pypi.org/project/langgraph-up-devkits/
- Test PyPI: https://test.pypi.org/project/langgraph-up-devkits/
- Packaging Guide: https://packaging.python.org/tutorials/packaging-projects/
- Twine Documentation: https://twine.readthedocs.io/