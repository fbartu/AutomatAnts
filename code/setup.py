
import sys
import distutils.core

sys.path.append('/home/polfer/.local/lib/python3.8/site-packages/')

import Cython.Build
distutils.core.setup(
    ext_modules = Cython.Build.cythonize("Run.pyx"))