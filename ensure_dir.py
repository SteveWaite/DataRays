import os


def ensure_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def ensure_dir_for_file(file_path):
    ensure_dir(os.path.dirname(file_path))
