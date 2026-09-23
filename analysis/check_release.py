#!/usr/bin/env python3
"""Check portable release files without loading models or accessing the network."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_ROOTS = {'.git', '.venv', 'venv', 'build', 'outputs', 'logs', 'models',
                  'checkpoints', 'captures', 'artifacts', 'local_data', 'inputs'}
EXCLUDED_PARTS = {'__pycache__', '.pytest_cache', '.ipynb_checkpoints'}
TENSORS = {'.pt', '.pth', '.bin', '.ckpt', '.safetensors', '.npz', '.npy'}
TEXT = {'.py', '.md', '.txt', '.csv', '.json', '.yaml', '.yml', '.cff', '.toml'}
MANIFEST = ROOT/'data/provenance/release_manifest.json'

# Match filesystem literals, while leaving URL schemes and relative links alone.
PATH_LITERAL = re.compile(r'^(?:~/|/|[A-Za-z]:[\\/])[A-Za-z_.][A-Za-z0-9_. /\\:-]*$')
TEXT_PATH = re.compile(r'(?<![\w:./\\])(?:/[A-Za-z_][A-Za-z0-9_.-]*/[A-Za-z0-9_.-]+|~/[A-Za-z0-9_.-]+|[A-Za-z]:[\\/][A-Za-z0-9_.-]+)')


def absolute_path_line(content, tree=None):
    """Find fixed filesystem paths, excluding separators and URL/regex fragments."""
    if tree is None:
        match = TEXT_PATH.search(content)
        return content.count('\n', 0, match.start())+1 if match else None
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        value = node.value
        parent = parents.get(node)
        # Suffixes in a URL/path expression are not standalone filesystem paths.
        if isinstance(parent, ast.JoinedStr) and parent.values[0] is not node:
            continue
        if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Add) and parent.right is node:
            continue
        if PATH_LITERAL.fullmatch(value):
            return node.lineno
        if isinstance(parent, ast.Expr):  # Also inspect docstrings.
            match = TEXT_PATH.search(value)
            if match:
                return node.lineno+value.count('\n', 0, match.start())
    return None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release_files():
    selected = []
    for path in ROOT.rglob('*'):
        rel = path.relative_to(ROOT)
        # Vendored EasyEdit/easyeditor/models contains source, not model weights.
        if rel.parts[0] in EXCLUDED_ROOTS or any(part in EXCLUDED_PARTS for part in rel.parts):
            continue
        if path.is_symlink():
            raise ValueError(f'Symlink is not a portable release file: {rel}')
        if not path.is_file() or path.suffix in {'.pyc', '.zip'}:
            continue
        if path.name.endswith(('.local.json', '.local.yaml')):
            continue
        selected.append(path)
    return sorted(selected)


def inspect():
    files = release_files()
    links = python_files = 0
    for path in files:
        name = str(path.relative_to(ROOT))
        if path.suffix in TENSORS or path.stat().st_size > 50_000_000:
            raise ValueError(f'Large experiment artifact in release: {name}')
        if path.name == '.env' or (path.name.startswith('.env.') and path.name != '.env.example'):
            raise ValueError(f'Local environment file in release: {name}')
        if path.suffix not in TEXT:
            continue
        text = path.read_text(encoding='utf-8')
        # Report locations only; never print potentially secret values.
        tree = ast.parse(text, filename=name) if path.suffix == '.py' else None
        path_line = absolute_path_line(text, tree)
        token = re.search(r'\b(?:hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{32,})\b', text)
        if path_line or token:
            line = path_line or text.count('\n', 0, token.start())+1
            reason = 'hardcoded absolute filesystem path' if path_line else 'credential-like literal'
            raise ValueError(f'{reason} in {name}:{line}')
        if path.suffix == '.py':
            python_files += 1
        if path.suffix == '.md':
            for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
                parsed = urlsplit(target)
                if parsed.scheme or target.startswith('#'):
                    continue
                dest = (path.parent/unquote(parsed.path)).resolve()
                if not dest.is_relative_to(ROOT) or not dest.exists():
                    raise ValueError(f'Broken local link: {name} -> {target}')
                links += 1
    return files, dict(passed=True, files=len(files), python_files=python_files,
                       local_links=links, total_bytes=sum(p.stat().st_size for p in files),
                       model_inference=False, network_access=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true',
                        help='Refresh the release snapshot after reviewing intended changes.')
    args = parser.parse_args()
    files, report = inspect()
    expected = {str(p.relative_to(ROOT)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                for p in files if p != MANIFEST}
    if args.write_manifest:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps({'schema': 1, 'files': expected}, indent=2)+'\n')
    else:
        if not MANIFEST.is_file():
            raise ValueError('Missing release manifest; use --write-manifest after reviewing changes.')
        saved = json.loads(MANIFEST.read_text())['files']
        if expected != saved:
            differences = sorted(k for k in expected.keys() | saved.keys() if expected.get(k) != saved.get(k))
            raise ValueError('Release differs from reviewed snapshot: '+', '.join(differences[:12]))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
