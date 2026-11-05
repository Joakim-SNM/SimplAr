"""Where the tools keep their files.

Results and sessions go to a data folder outside the program folder, one
subfolder per tool:

    SimplAr data/
        1 Data converter/
        2 Flux monitors/
        3 Flux regression/
        4 Plateau & isochron/
        5 Age distribution/

The data folder defaults to "SimplAr data" in the user's Documents folder and
can be changed from the launcher (stored with QSettings, so it is the same for
every tool).  The environment variable SIMPLAR_DATA_DIR overrides it; the tests
use that to work in a temporary folder.

The get_*_file_name functions take the same arguments as the QFileDialog
functions they wrap.  Each dialog opens in its tool's folder, or in the folder
last used for that tool when that lies inside the data folder; a suggested file
name is made safe for Windows.  Settings and caches that are not results (figure
export presets, rendered equations) go to the user's configuration and cache
folders.
"""

import os
import re

from PyQt5.QtCore import QSettings, QStandardPaths
from PyQt5.QtWidgets import QFileDialog

TOOL_FOLDERS = {
    'data_converter':   '1 Data converter',
    'flux_monitors':    '2 Flux monitors',
    'flux_regression':  '3 Flux regression',
    'plateau_isochron': '4 Plateau & isochron',
    'age_distribution': '5 Age distribution',
}

_current_tool = None


def use_tool(tool):
    """Set the tool whose folder the dialogs of this process open in."""
    global _current_tool
    if tool not in TOOL_FOLDERS:
        raise ValueError(f'unknown tool {tool!r}')
    _current_tool = tool


def _settings():
    return QSettings('SimplAr', 'SimplAr')


def default_data_root():
    docs = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
    return os.path.join(docs or os.path.expanduser('~'), 'SimplAr data')


def data_root():
    """The data folder (not created here)."""
    env = os.environ.get('SIMPLAR_DATA_DIR')
    if env:
        return env
    v = _settings().value('dataRoot', '', type=str)
    return v or default_data_root()


def set_data_root(path):
    _settings().setValue('dataRoot', os.path.abspath(path))


def tool_dir(tool=None, create=True):
    """The tool's subfolder of the data folder (the data folder itself when no
    tool is set)."""
    tool = tool or _current_tool
    d = data_root() if tool is None else os.path.join(data_root(), TOOL_FOLDERS[tool])
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def create_all():
    for tool in TOOL_FOLDERS:
        tool_dir(tool)
    return data_root()


def _inside(path, root):
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) \
            == os.path.abspath(root)
    except ValueError:            # different drives on Windows
        return False


def start_dir(tool=None):
    """Folder a dialog opens in: the last one used for this tool if it is still
    inside the data folder, else the tool's own folder."""
    tool = tool or _current_tool
    last = _settings().value(f'lastDir/{tool or "root"}', '', type=str)
    if last and os.path.isdir(last) and _inside(last, data_root()):
        return last
    return tool_dir(tool)


def remember(path, tool=None):
    tool = tool or _current_tool
    if not path:
        return
    d = path if os.path.isdir(path) else os.path.dirname(path)
    if d and _inside(d, data_root()):
        _settings().setValue(f'lastDir/{tool or "root"}', d)


_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_filename(name, fallback='untitled'):
    """A file name valid on Windows, macOS and Linux: forbidden characters
    become '_', trailing dots and spaces are dropped."""
    s = _UNSAFE.sub('_', str(name)).strip().rstrip('. ')
    return s or fallback


def _start_path(directory, tool):
    """The directory argument of a dialog, resolved against the tool's folder.
    An empty value or a bare file name starts in start_dir(); a path that
    already names a folder is kept."""
    directory = directory or ''
    if os.path.isdir(directory):
        return directory
    head, tail = os.path.split(directory)
    if head and os.path.isdir(head):
        return os.path.join(head, safe_filename(tail))
    # anything else is a suggested file name ('/' and ':' in it included)
    base = start_dir(tool)
    return os.path.join(base, safe_filename(directory)) if directory else base


def get_open_file_name(parent, caption='', directory='', filter='', tool=None, **kw):
    path, flt = QFileDialog.getOpenFileName(parent, caption, _start_path(directory, tool),
                                            filter, **kw)
    remember(path, tool)
    return path, flt


def get_open_file_names(parent, caption='', directory='', filter='', tool=None, **kw):
    paths, flt = QFileDialog.getOpenFileNames(parent, caption, _start_path(directory, tool),
                                              filter, **kw)
    if paths:
        remember(paths[0], tool)
    return paths, flt


def get_save_file_name(parent, caption='', directory='', filter='', tool=None, **kw):
    path, flt = QFileDialog.getSaveFileName(parent, caption, _start_path(directory, tool),
                                            filter, **kw)
    remember(path, tool)
    return path, flt


def get_existing_directory(parent, caption='', directory='', tool=None, **kw):
    path = QFileDialog.getExistingDirectory(parent, caption, _start_path(directory, tool), **kw)
    remember(path, tool)
    return path


def config_path(filename):
    """A settings file in the user's configuration folder (SimplAr/filename)."""
    base = QStandardPaths.writableLocation(QStandardPaths.GenericConfigLocation)
    d = os.path.join(base or os.path.expanduser('~'), 'SimplAr')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)


def cache_dir(name):
    """A cache folder in the user's cache location (SimplAr/name)."""
    base = QStandardPaths.writableLocation(QStandardPaths.GenericCacheLocation)
    d = os.path.join(base or os.path.expanduser('~'), 'SimplAr', name)
    os.makedirs(d, exist_ok=True)
    return d
