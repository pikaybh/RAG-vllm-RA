import os
import logging
from typing import Callable, Optional, Dict

from langchain_core.retrievers import BaseRetriever

from templates import kras_map


def mapper(map: Dict[str, str], *args) -> str:
    valid_items = [f"{key}: {value}" for key, value in map.items() if value in args]
    return "\n".join(valid_items)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

__all__ = ["mapper", "format_docs"]