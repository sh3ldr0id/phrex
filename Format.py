from os import listdir, remove
from uuid import uuid1
from tqdm import tqdm

source_dir = "Z:/Dev/Gen4/Phrex/Dataset/unformatted/media/"
dest_dir = "Z:/Dev/Gen4/Phrex/Dataset/unformatted/mixed/"

files = listdir(source_dir)

for file_name in tqdm(files):
    source_file_path = source_dir + file_name
    dest_file_path = dest_dir + str(uuid1()) + ".jpg"

    with open(source_file_path, "rb") as source_file:
        data = source_file.read()

    with open(dest_file_path, "wb") as dest_file:
        dest_file.write(data)

    remove(source_file_path)