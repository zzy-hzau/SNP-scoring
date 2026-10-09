"""Path and external-tool I/O shared by the two source workflows."""
import gzip
import json
import os
from pathlib import Path
import shlex
import subprocess
import urllib.request


def configuration(path, output):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    config['_directory'] = path.parent
    config['_output'] = Path(output).resolve()
    config['_output'].mkdir(parents=True, exist_ok=True)
    return config


def input_path(config, value):
    path = Path(os.path.expandvars(str(value)))
    return path if path.is_absolute() else config['_directory'] / path


def open_text(path):
    return gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)


def acquire(config, item, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        return destination
    if item.get('path'):
        source = input_path(config, item['path'])
        return source
    urllib.request.urlretrieve(item['url'], destination)
    return destination


def command(config, argv, *, stdout=None, dry_run=False):
    argv = [str(value) for value in argv]
    argv[0] = config.get('tools', {}).get(argv[0], argv[0])
    print(shlex.join(argv), flush=True)
    if dry_run:
        return
    if stdout is not None:
        with open(stdout, 'w') as handle:
            subprocess.run(argv, stdout=handle, check=True)
    else:
        subprocess.run(argv, check=True)
    with (config['_output'] / 'commands.jsonl').open('a') as handle:
        handle.write(json.dumps({'argv': argv, 'stdout': str(stdout) if stdout else None})+'\n')


def plain_file(source, destination):
    with open_text(source) as fi, open(destination, 'w') as fo:
        for line in fi:
            fo.write(line)
