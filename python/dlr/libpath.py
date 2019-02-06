# coding: utf-8
"""Find the path to DLR dynamic library files."""

import os
import platform
import sys

class DLRLibraryNotFound(Exception):
    """Error thrown by when DLR is not found"""
    pass


def find_lib_path():
    """Find the path to DLR dynamic library files.

    Returns
    -------
    lib_path: list(string)
       List of all found library path to DLR
    """
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    dll_path = [curr_path, os.path.join(curr_path, '../../lib/'),
                os.path.join(curr_path, './lib/'),
                os.path.join(sys.prefix, 'dlr'),
                os.path.join(sys.prefix, 'local', 'dlr'),
                os.path.join(sys.exec_prefix, 'local', 'dlr'),
                os.path.join(os.path.expanduser('~'), '.local', 'dlr')]
    if sys.platform == 'win32':
        if platform.architecture()[0] == '64bit':
            dll_path.append(os.path.join(curr_path, '../../windows/x64/Release/'))
            # hack for pip installation when copy all parent source directory here
            dll_path.append(os.path.join(curr_path, './windows/x64/Release/'))
        else:
            dll_path.append(os.path.join(curr_path, '../../windows/Release/'))
            # hack for pip installation when copy all parent source directory here
            dll_path.append(os.path.join(curr_path, './windows/Release/'))
        dll_path = [os.path.join(p, 'dlr.dll') for p in dll_path]
    elif sys.platform.startswith('linux') or sys.platform.startswith('freebsd'):
        dll_path = [os.path.join(p, 'libdlr.so') for p in dll_path]
    elif sys.platform == 'darwin':
        dll_path = [os.path.join(p, 'libdlr.dylib') for p in dll_path]

    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]

    if not lib_path and not os.environ.get('DLR_BUILD_DOC', False):
        raise DLRLibraryNotFound(
            'Cannot find DLR Library in the candidate path, ' +
            'List of candidates:\n' + ('\n'.join(dll_path)))
    return lib_path
