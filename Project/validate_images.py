import os
import csv
import hashlib
import re
from PIL import Image
import shutil

def validate_images(input_dir: str, output_dir: str,
                    log_file: str, formatter: str = "07d"):

    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        raise ValueError(f"{input_dir} is not an existing directory")

    os.makedirs(output_dir, exist_ok=True)

    counter = 0
    copied_files = set()

    with open("labels.csv", "a", newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';')
        csvwriter.writerow(["name", "label"])

    with open(log_file, 'w') as log:
        for root, _, files in os.walk(input_dir):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                rel_file_path = os.path.relpath(file_path, input_dir)

                #check for error
                if check_error(file_path, copied_files):
                    error_num = check_error(file_path, copied_files)
                    log.write(f"{rel_file_path},{error_num}\n")
                else:
                    counter += 1
                    new_filename = "{:0{}}.jpg".format(counter-1, formatter)
                    shutil.copy(file_path, os.path.join(output_dir, new_filename))
                    copied_files.add(hash_func(file_path))
                    with open("labels.csv", "a", newline='') as csvfile:
                        csvwriter = csv.writer(csvfile, delimiter=';')
                        label = re.sub(r'\d+', '', os.path.splitext(file)[0])
                        csvwriter.writerow([new_filename, label])

    return counter

def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
            return True
    except Exception:
        return False

def has_valid_dimensions(file_path):
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            return width >= 100 and height >= 100 and (img.mode == "RGB" or img.mode == "L")
    except Exception:
        return False

def has_variance(file_path):
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            pixels = img.load()
            variance = sum(sum(pixels[x, y]) for x in range(width) for y in range(height)) / (width * height)
            return variance > 0
    except Exception:
        return False

def is_duplicate(file_path, copied_files):
    return hash_func(file_path) in copied_files

def hash_func(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(65536)  # 64KB buffer
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()

def check_error(file_path, copied_files):
    if not file_path.lower().endswith(('.jpg', '.jpeg')):
        return "1"
    elif os.path.getsize(file_path) > 250000:
        return "2"
    elif not is_valid_image(file_path):
        return "3"
    elif not has_valid_dimensions(file_path):
        return "4"
    elif not has_variance(file_path):
        return "5"
    elif is_duplicate(file_path, copied_files):
        return "6"
