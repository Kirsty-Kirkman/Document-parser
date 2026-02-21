# Document-parser
Parser tool for well integrity data

### Code Structure

```python
src/
│
├── __init__.py
├── image/   # contains code to process and structure diagrams
│   ├── __init__.py
│   ├── extract.py
│   └── preprocess.py
├── text/    # contains code to extract and parse text/annotations
│   ├── __init__.py
│   └── extract_text.py
|
examples/   # sample images to try the workflow on
```

### Start Here

Install `pixi`

For Windows
```bash
powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

For Mac
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

```bash
pixi install

# Run pipeline on examples
pixi run python main.py
```