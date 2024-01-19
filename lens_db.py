import exiftool
import lensfunpy

src = 'DSCF7627.RAF'

with exiftool.ExifToolHelper() as et:
    metadata = et.get_metadata(src)[0]


CAMERA_DB = {'X100S': {'cam_make': 'Fujifilm', 'cam_model': 'X100S', 'lens_make': 'Fujifilm', 'lens_model': 'X100 & compatibles (Standard)'}}

db = lensfunpy.Database()
try:
    cam = db.find_cameras(metadata['EXIF:Make'], metadata['EXIF:Model'], loose_search=True)[0]
    lens = db.find_lenses(cam, metadata['EXIF:LensMake'], metadata['EXIF:LensModel'], loose_search=True)[0]
except KeyError:
    try:
        camera = metadata['EXIF:Model']
        cam = db.find_cameras(CAMERA_DB[camera]['cam_make'], CAMERA_DB[camera]['cam_model'], loose_search=True)[0]
        lens = db.find_lenses(cam, CAMERA_DB[camera]['lens_make'], CAMERA_DB[camera]['lens_model'], loose_search=True)[0]
    except KeyError:
        print('megafail')
try:
    lens = metadata['EXIF:FocalLength']
    aperture = metadata['EXIF:ApertureValue']


except:
    print('megafail')