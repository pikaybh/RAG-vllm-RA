import os

def extfomatter(ext: str) -> str:
    ext = ext.lower()
    if not ext.startswith('.'):
        ext = f".{ext}"
    return ext


def isext(path: str, *args) -> bool:
    return os.path.splitext(path)[-1] in [extfomatter(ext) for ext in args]

__all__ = ["isext"]