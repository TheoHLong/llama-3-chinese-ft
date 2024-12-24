# Development Guide

## Project Structure

```
llama-3-chinese-ft/
├── README.md                  # Project overview
├── requirements.txt           # Dependencies
├── notebooks/                 # Jupyter notebooks
├── src/                      # Source code
│   ├── data_preparation.py   # Data processing
│   ├── model_training.py     # Training logic
│   ├── inference.py          # Inference code
│   └── utils.py             # Utilities
├── configs/                  # Configuration files
├── scripts/                  # Shell scripts
├── tests/                    # Unit tests
└── docs/                     # Documentation
```

## Development Workflow

1. Code Style
   - Follow PEP 8 guidelines
   - Use type hints
   - Add docstrings to functions
   - Format code with black
   - Sort imports with isort

2. Testing
   ```bash
   # Run tests
   pytest tests/

   # Run with coverage
   pytest --cov=src tests/
   ```

3. Documentation
   - Update relevant .md files
   - Add docstrings to new functions
   - Keep README.md updated

## Contributing

1. Creating Pull Requests
   - Fork the repository
   - Create feature branch
   - Make changes
   - Run tests
   - Submit PR

2. Code Review Process
   - Code style check
   - Test coverage
   - Documentation review
   - Performance impact

3. Best Practices
   - Keep PRs focused and small
   - Write descriptive commit messages
   - Update tests for new features
   - Document significant changes

## Versioning

We use semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Incompatible API changes
- MINOR: Backwards-compatible features
- PATCH: Backwards-compatible fixes