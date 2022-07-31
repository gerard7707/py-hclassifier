import os
from os.path import join, dirname

PROJECT_ROOT = dirname(__file__)

DIR_MODEL = join(PROJECT_ROOT, "model")

PATH_RAINBOW_YAML = join(DIR_MODEL, "rainbow.yaml")

PATH_OBJECTS_YAML = join(PROJECT_ROOT, "objects.yaml")

for _pending in [DIR_MODEL]:
    os.makedirs(_pending, exist_ok=True)
