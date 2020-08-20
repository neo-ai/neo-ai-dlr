# coding: utf-8
"""Find the path to DLR dynamic library files."""

import os
import platform
import sys

class DLRLibraryNotFound(Exception):
    """Error thrown by when DLR is not found"""
    pass


def find_lib_path(model_path=None, use_default_dlr=True, logger=None, setup=False):
    """Find the path to DLR dynamic library files."""

    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    if setup:
        # Only look in build directory when installing or building wheel.
        dll_paths = [os.path.join(curr_path, '../../build/lib/')]
    else:
        # Prioritize library in system path over the current_path.
        dll_paths = [os.path.join(sys.prefix, 'dlr'),
                     os.path.join(sys.prefix, 'local', 'dlr'),
                     os.path.join(sys.exec_prefix, 'local', 'dlr'),
                     os.path.join(os.path.expanduser('~'), '.local', 'dlr'),
                     os.path.join(curr_path, '../../lib/'),
                     os.path.join(curr_path, '../../build/lib/'),
                     os.path.join(curr_path, './lib/'),
                     os.path.join(curr_path, './build/lib/'), 
                     curr_path]
    
    if sys.platform == 'win32':
        if platform.architecture()[0] == '64bit':
            dll_paths.append(os.path.join(curr_path, '../../windows/x64/Release/'))
            # hack for pip installation when copy all parent source directory here
            dll_paths.append(os.path.join(curr_path, './windows/x64/Release/'))
        else:
            dll_paths.append(os.path.join(curr_path, '../../windows/Release/'))
            # hack for pip installation when copy all parent source directory here
            dll_paths.append(os.path.join(curr_path, './windows/Release/'))
        libname = 'dlr.dll'
    elif sys.platform.startswith('linux') or sys.platform.startswith('freebsd'):
        libname = 'libdlr.so'
    elif sys.platform == 'darwin':
        libname = 'libdlr.dylib'

    if model_path is not None:
        libpath = os.path.join(model_path, libname)
        if not use_default_dlr and os.path.exists(libpath):
            if logger:
                logger.info("Found {} in model artifact. Using dlr from {}".format(libname, libpath))
            return libpath

    dll_paths = [os.path.join(p, libname) for p in dll_paths]

    lib_paths = [p for p in dll_paths if os.path.exists(p) and os.path.isfile(p)]

    if not lib_paths and not os.environ.get('DLR_BUILD_DOC', False):
        raise DLRLibraryNotFound(
            'Cannot find DLR Library in the candidate path, ' +
            'List of candidates:\n' + ('\n'.join(dll_paths)))

    if logger is not None:
        if use_default_dlr:
            logger.info("Forced to use default {} from {}".format(libname, lib_paths[0]))
        else:
            logger.info("Could not find {} in model artifact. Using dlr from {}".format(libname, lib_paths[0]))

    # If multiple paths are found always prefer the one that is in system paths.
    return lib_paths[0]
