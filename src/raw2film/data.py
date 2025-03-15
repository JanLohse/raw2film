import numpy as np

METADATA_KEYS = ['GPSDateStamp', 'ModifyDate', 'FocalLengthIn35mmFormat', 'ShutterSpeedValue', 'FocalLength', 'Make',
    'Saturation', 'SubSecTimeOriginal', 'SubSecTimeDigitized', 'GPSImgDirectionRef', 'ExposureProgram',
    'GPSLatitudeRef', 'Software', 'GPSVersionID', 'ResolutionUnit', 'LightSource', 'FileSource', 'ExposureMode',
    'Compression', 'MaxApertureValue', 'OffsetTime', 'DigitalZoomRatio', 'Contrast', 'InteropIndex', 'ThumbnailLength',
    'DateTimeOriginal', 'OffsetTimeOriginal', 'SensingMethod', 'SubjectDistance', 'CreateDate', 'ExposureCompensation',
    'SensitivityType', 'ApertureValue', 'ExifImageWidth', 'SensorLeftBorder', 'FocalPlaneYResolution',
    'GPSImgDirection', 'ComponentsConfiguration', 'Flash', 'Model', 'ColorSpace', 'LensModel', 'XResolution',
    'GPSTimeStamp', 'ISO', 'CompositeImage', 'FocalPlaneXResolution', 'SubSecTime', 'GPSAltitude',
    'OffsetTimeDigitized', 'ExposureTime', 'LensMake', 'WhiteBalance', 'BrightnessValue', 'GPSLatitude', 'YResolution',
    'GPSLongitude', 'YCbCrPositioning', 'Copyright', 'SubjectDistanceRange', 'SceneType', 'GPSAltitudeRef',
    'FocalPlaneResolutionUnit', 'MeteringMode', 'GPSLongitudeRef', 'SensorTopBorder', 'SceneCaptureType', 'FNumber',
    'LightValue', 'BrightnessValue', 'SensorWidth', 'SensorHeight', 'SensorBottomBorder', 'SensorRightBorder',
    'ProcessingSoftware']
EXTENSION_LIST = ('.rw2', '.dng', '.crw', '.cr2', '.cr3', '.nef', '.orf', '.ori', '.raf', '.rwl', '.pef', '.ptx')
FORMATS = {'110': (17, 13),
           '135-half': (24, 18), '135': (36, 24),
           'xpan': (65, 24),
           '120-4.5': (56, 42), '120-6': (56, 56), '120': (70, 56), '120-9': (83, 56),
           '4x5': (127, 101.6), '5x7': (177.8, 127), '8x10': (254, 203.2), '11x14': (355.6, 279.4),
           'super16': (12.42, 7.44), 'scope': (24.89, 10.4275), 'flat': (24.89, 13.454), 'academy': (24.89, 18.7),
           '65mm': (48.56, 22.1), 'IMAX': (70.41, 52.63)}
REC2020_TO_ARRIWCG = np.array([[1.0959, -.0751, -.0352],
                               [-.1576, 0.8805, 0.0077],
                               [0.0615, 0.1946, 1.0275]], dtype=np.float32)
REC2020_TO_REC709 = np.array([[1.6605, -.1246, -.0182],
                              [-.5879, 1.1330, -.1006],
                              [-.0728, -.0084, 1.1187]], dtype=np.float32)
REC709_TO_REC2020 = np.array([[.6274, .0691, .0164],
                              [.3294, .9195, .0880],
                              [.0433, .0114, .8956]], dtype=np.float32)
REC2020_TO_XYZ = np.array([[0.6369580, 0.1446169, 0.1688810],
                           [0.2627002, 0.6779981, 0.0593017],
                           [0.0000000, 0.0280727, 1.0609851]], dtype=np.float32)
XYZ_TO_REC2020 = np.array([[1.7166512, -0.3556708, -0.2533663],
                           [-0.6666844, 1.6164812, 0.0157685],
                           [0.0176399, -0.0427706, 0.9421031]], dtype=np.float32)
CAMERA_DB = {"X100S": "Fujifilm : X100S",
             "DMC-GX80": "Panasonic : DMC-GX80",
             "DC-FZ10002": "Panasonic : DC-FZ10002"}
LENS_DB = {"X100S": "Fujifilm : X100 & compatibles (Standard)",
           "LUMIX G 25/F1.7": "Panasonic : Lumix G 25mm f/1.7 Asph.",
           "LUMIX G VARIO 12-32/F3.5-5.6": "Panasonic : Lumix G Vario 12-32mm f/3.5-5.6 Asph. Mega OIS",
           "DC-FZ10002": "Leica : FZ1000 & compatibles"}
FILM_DB = {"250D": {'r_a': 1.020, 'r_f': 34, 'g_a': 1.034, 'g_f': 52, 'b_a': 1.064, 'b_f': 63,
                    'rough': [10, 11, 17], 'clean': [4, 5, 6]},
           "500T": {'r_a': 1.073, 'r_f': 60, 'g_a': 1.052, 'g_f': 53, 'b_a': 1.039, 'b_f': 34,
                    'rough': [10, 14, 31], 'clean': [5, 6, 10]},
           "200T": {'r_a': 1.008, 'r_f': 38, 'g_a': 1.092, 'g_f': 56, 'b_a': 1.120, 'b_f': 65,
                    'rough': [9, 10, 23], 'clean': [4, 5, 12]},
           "50D": {'r_a': 1.023, 'r_f': 36, 'g_a': 1.024, 'g_f': 44, 'b_a': 1.000, 'b_f': 36,
                   'rough': [7, 7, 13], 'clean': [2, 3, 5]}}
