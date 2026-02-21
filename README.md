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

```bash
pixi install
```

### Usage Instructions

```bash
python src.image.FUNCTION exampes/IMAGE.png
python src.text.FUNCTON examples/IMAGE.png
```
where FUNCTION is the step that you want to run on the IMAGE