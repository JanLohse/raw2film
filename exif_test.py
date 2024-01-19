import os

import exiftool

with exiftool.ExifToolHelper() as et:
    metadata = et.get_metadata("PXL_20240119_120108269.jpg")[0]

print(metadata.keys())