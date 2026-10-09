"""Locate caller resources without embedding an author checkout path."""
import os
from pathlib import Path

def project_root(module_file=None):
    if os.environ.get('PHENO_PROJECT_ROOT'):
        return Path(os.environ['PHENO_PROJECT_ROOT']).resolve()
    if module_file:
        for directory in Path(module_file).resolve().parents:
            if (directory / 'pyproject.toml').is_file() and (directory / 'configs').is_dir():
                return directory
    return Path.cwd()

def package_dir(domain=''):
    return Path(__file__).resolve().parents[1] / domain
