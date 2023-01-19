from typing import Dict, List, Tuple
import platform
from os import path, remove, walk
from subprocess import run, Popen
import yaml
import zipfile

CONFIG_FILE: str = "config.yaml"
if platform.system() == "Windows":
    FILETYPES_TO_IGNORE: Tuple = (".c", ".so", "pyx")
    ROOT_DIRECTORY = "../"
    ZIP_FILES: List[str] = [
        "config.yaml",
        "ladder.py",
        "run.py",
    ]
else:
    FILETYPES_TO_IGNORE: Tuple = (".c", ".pyd", "pyx")
    ROOT_DIRECTORY = "./"
    ZIP_FILES: List[str] = [
        "config.yaml",
        "ladder.py",
        "run.py",
    ]

ZIPFILE_NAME: str = "kitten.zip"

ZIP_DIRECTORIES: Dict[str, Dict] = {
    "bot": {"zip_all": True, "folder_to_zip": "bot"},
    "python-sc2": {"zip_all": False, "folder_to_zip": "sc2"},
    "MapAnalyzer": {"zip_all": True, "folder_to_zip": "MapAnalyzer"},
}


def zip_dir(dir_path, zip_file):
    """
    Will walk through a directory recursively and add all folders and files to zipfile
    @param dir_path:
    @param zip_file:
    @return:
    """
    for root, _, files in walk(dir_path):
        for file in files:
            if file.lower().endswith(FILETYPES_TO_IGNORE):
                continue
            zip_file.write(
                path.join(root, file),
                path.relpath(path.join(root, file), path.join(dir_path, "..")),
            )


def zip_files_and_directories() -> None:
    """
    @return:
    """

    path_to_zipfile = path.join(ROOT_DIRECTORY, ZIPFILE_NAME)
    # if the zip file already exists remove it
    if path.isfile(path_to_zipfile):
        remove(path_to_zipfile)
    # create a new zip file
    zip_file = zipfile.ZipFile(path_to_zipfile, "w", zipfile.ZIP_DEFLATED)

    # write directories to the zipfile
    for directory, values in ZIP_DIRECTORIES.items():
        if values["zip_all"]:
            zip_dir(path.join(ROOT_DIRECTORY, directory), zip_file)
        else:
            path_to_dir = path.join(ROOT_DIRECTORY, directory, values["folder_to_zip"])
            zip_dir(path_to_dir, zip_file)

    # write individual files
    for single_file in ZIP_FILES:
        zip_file.write(path.join(ROOT_DIRECTORY, single_file), single_file)
    # close the zip file
    zip_file.close()


if __name__ == "__main__":
    zip_files_and_directories()

