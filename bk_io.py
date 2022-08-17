from ensure_dir import ensure_dir_for_file
import os
from typing import Iterable


def write_text_file(file_path: os.PathLike, content: str, open_mode: str = 'w') -> None:
    ensure_dir_for_file(file_path)
    with open(file_path, open_mode) as f:
        f.write(content)


def write_binary_file(file_path: os.PathLike, content: bytes, open_mode: str = 'wb') -> None:
    ensure_dir_for_file(file_path)
    with open(file_path, open_mode) as f:
        f.write(content)


def write_lines_to_file(file_path: os.PathLike, lines: Iterable[str]):
    ensure_dir_for_file(file_path)
    write_text_file(file_path, '\n'.join(lines))


def read_text_file(file_path: os.PathLike, open_mode: str = 'r') -> str:
    with open(file_path, open_mode) as f:
        return f.read()


def read_binary_file(file_path: os.PathLike, open_mode: str = 'rb') -> bytes:
    with open(file_path, open_mode) as f:
        return f.read()
