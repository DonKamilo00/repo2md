# Repo2MD [WIP]📂→📝

A small tool to convert entire GitHub repositories into well-structured Markdown documentation with preserved code formatting, directory structure, and metadata.

#### Note: This tool is still in experimental stage it might not work correctly for some of the repos


[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Features ✨

- **Full Repository Conversion**: Convert all text-based files in a repo to Markdown
- **Directory Structure Visualization**: Auto-generated ASCII directory tree
- **Smart File Handling**:
  - Exclude specific directories (`.git`, `node_modules`, etc.)
  - Skip binary files automatically
  - Configurable file size limits
- **Rich Metadata**:
  - Repository description
  - Owner information
  - Stars/Forks count
  - Creation/Update dates
  - License information
- **CLI Interface**: Easy-to-use command line arguments
- **Syntax Highlighting**: Automatic language detection for code blocks

## Installation ⚙️

### Prerequisites
- Python 3.8+
- Git

### Install from PyPI
```bash
pip install repo2md
