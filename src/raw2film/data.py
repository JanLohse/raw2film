import numpy as np
from spectral_film_lut import FILMSTOCKS

METADATA_KEYS = ['GPSDateStamp', 'ModifyDate', 'FocalLengthIn35mmFormat', 'ShutterSpeedValue', 'FocalLength', 'Make',
                 'Saturation', 'SubSecTimeOriginal', 'SubSecTimeDigitized', 'GPSImgDirectionRef', 'ExposureProgram',
                 'GPSLatitudeRef', 'Software', 'GPSVersionID', 'ResolutionUnit', 'LightSource', 'FileSource',
                 'ExposureMode', 'Compression', 'MaxApertureValue', 'OffsetTime', 'DigitalZoomRatio', 'Contrast',
                 'InteropIndex', 'ThumbnailLength', 'DateTimeOriginal', 'OffsetTimeOriginal', 'SensingMethod',
                 'SubjectDistance', 'CreateDate', 'ExposureCompensation', 'SensitivityType', 'ApertureValue',
                 'ExifImageWidth', 'SensorLeftBorder', 'FocalPlaneYResolution', 'GPSImgDirection',
                 'ComponentsConfiguration', 'Flash', 'Model', 'ColorSpace', 'LensModel', 'XResolution', 'GPSTimeStamp',
                 'ISO', 'CompositeImage', 'FocalPlaneXResolution', 'SubSecTime', 'GPSAltitude', 'OffsetTimeDigitized',
                 'ExposureTime', 'LensMake', 'WhiteBalance', 'BrightnessValue', 'GPSLatitude', 'YResolution',
                 'GPSLongitude', 'YCbCrPositioning', 'Copyright', 'SubjectDistanceRange', 'SceneType', 'GPSAltitudeRef',
                 'FocalPlaneResolutionUnit', 'MeteringMode', 'GPSLongitudeRef', 'SensorTopBorder', 'SceneCaptureType',
                 'FNumber', 'LightValue', 'BrightnessValue', 'SensorWidth', 'SensorHeight', 'SensorBottomBorder',
                 'SensorRightBorder', 'ProcessingSoftware']
EXTENSION_LIST = ('.rw2', '.dng', '.crw', '.cr2', '.cr3', '.nef', '.orf', '.ori', '.raf', '.rwl', '.pef', '.ptx')
FORMATS = {'110': (17, 13),
           '135-half': (24, 18), '135': (36, 24),
           'xpan': (65, 24),
           '120-4.5': (56, 42), '120-6': (56, 56), '120': (70, 56), '120-9': (83, 56),
           '4x5': (127, 101.6), '5x7': (177.8, 127), '8x10': (254, 203.2), '11x14': (355.6, 279.4),
           'super16': (12.42, 7.44), 'scope': (24.89, 10.4275), 'flat': (24.89, 13.454), 'academy': (24.89, 18.7),
           '65mm': (48.56, 22.1), 'IMAX': (70.41, 52.63)}
REC709_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
XYZ_TO_REC709 = np.array([[3.2404542, -1.5371385, -0.4985314],
                          [-0.9692660, 1.8760108, 0.0415560],
                          [0.0556434, -0.2040259, 1.0572252]], dtype=np.float32)
CAMERA_DB = {"X100S": "Fujifilm : X100S",
             "DMC-GX80": "Panasonic : DMC-GX80",
             "DC-FZ10002": "Panasonic : DC-FZ10002"}
LENS_DB = {"X100S": "Fujifilm : X100 & compatibles (Standard)",
           "LUMIX G 25/F1.7": "Panasonic : Lumix G 25mm f/1.7 Asph.",
           "LUMIX G VARIO 12-32/F3.5-5.6": "Panasonic : Lumix G Vario 12-32mm f/3.5-5.6 Asph. Mega OIS",
           "DC-FZ10002": "Leica : FZ1000 & compatibles"}