import os
import sys
import waveweaver_app

from cx_Freeze import setup, Executable
sys.setrecursionlimit(5000)

if sys.platform == "win32":
    base = "gui"
else:
    base = None

executables=[Executable("waveweaver_app.py", base=base, icon=os.path.join('resources', 'ww_icon.ico'))]
include_files = ['resources/']

options= {
    "build_exe": {
        'include_files': include_files,
        'excludes': ['unittest'],
        'packages': ['OpenGL'],
        "zip_include_packages": ['PySide6', 'numpy'],
    },
}

setup(
    name='WaveWeaver',
    version=waveweaver_app.__version__,
    options=options,
    executables=executables,
)