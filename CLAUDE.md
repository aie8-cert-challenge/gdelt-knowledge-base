# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a certification challenge project for AI Engineering Bootcamp Cohort 8. It's a minimal Python starter project using `uv` for dependency management.

## Python Environment

- **Python Version**: 3.11 (specified in `.python-version`)
- **Package Manager**: `uv` for dependency management
- **Dependencies**: Managed via `pyproject.toml`

### Setup

```bash
# Create and activate virtual environment
uv venv --python 3.11
source .venv/bin/activate  # Linux/WSL/Mac
# Or on Windows:
# .venv\Scripts\activate

# Install dependencies (as they are added)
uv pip install -e .
```

### Running the Application

```bash
# Run the main script
python main.py
```

## Project Structure

- `main.py` — Entry point with main() function
- `pyproject.toml` — Python project configuration and dependencies
- `.python-version` — Specifies Python 3.11 requirement

## Development Workflow

This project uses the same git workflow as the parent AIE8 repository:
- Work on feature/assignment branches
- Do not commit directly to main
- Follow standard Python .gitignore patterns (already configured)
