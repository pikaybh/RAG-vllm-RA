import os

ext_map = {
    "pdf": ("pdf", "PDF"),
    "csv": ("csv", "CSV"),
    "json": ("json", "JSON"),
}

def extfomatter(ext: str) -> str:
    ext = ext.lower()
    return ext if ext.startswith('.') else "." + ext


def isext(path: str, *args) -> bool:
    return os.path.splitext(path)[-1] in [extfomatter(ext) for ext in args]

__all__ = ["extfomatter", "isext", "ext_map"]