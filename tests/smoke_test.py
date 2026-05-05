import os

from raw2film.__main__ import run

os.environ["QT_QPA_PLATFORM"] = "offscreen"
run(exit_immediately=True)
