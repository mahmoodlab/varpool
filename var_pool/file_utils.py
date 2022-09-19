import os
from glob import glob
import pathlib


def join_and_make(a, *p):
    """
    new_dir = os.path.join(a, *p)
    os.makedirs(new_dir, exist_ok=True)
    """
    new_dir = os.path.join(a, *p)
    os.makedirs(new_dir, exist_ok=True)
    return new_dir


def find_fpaths(folders, ext=['h5', 'pt'], names=None):
    """
    Gets all file paths from a folder or set of folders.

    Parameters
    ----------
    folders: str, list of str
        The folder or folders to check.

    ext: str or list of str
        Only get files ending in this extension.

    names: None, str
        (Optional) Subset of files to include in this dataset.

    Output
    ------
    fpaths: list of str
        The discovered file names ordered alphabetically by file name.
    """
    if isinstance(ext, str):
        ext = [ext]

    # format to list of str
    fpaths = []
    if isinstance(folders, str):
        folders = [folders]

    # find all available files with given extension
    for fd in folders:
        for e in ext:
            # find files in each feature directory
            fps = glob(os.path.join(fd, '*.{}'.format(e)))
            fpaths.extend(fps)

    # maybe subset to user specified file names
    if names is not None:
        fpaths = check_guest_list(restr=names, avail_fpaths=fpaths,
                                  drop_ext=True)

    # sort files alphabetically
    fpaths = sorted(fpaths, key=lambda p: os.path.basename(p))

    return fpaths


def check_guest_list(restr, avail_fpaths, drop_ext=False):
    """
    Returns a list of file paths from available files given a subset of file names to restrict ourselves to.

    Parameters
    ----------
    restr: list of str
        Names of the files we want to restrict ourselves to i.e. a guest list.

    avail_fpaths: list of str
        List of available file paths.

    drop_ext: bool
        Whether or not to drop the extension from the file paths in avail_fpaths.

    Output
    ------
    fpaths: list of str
        A subset of avail_fpaths who were on the guest list.
    """
    # get file names in case a full path was provided
    restr_names = set([os.path.basename(p) for p in restr])

    fpaths_ret = []
    for fpath in avail_fpaths:
        # pull out name of this available file
        if drop_ext:
            name = pathlib.Path(fpath).stem
        else:
            name = os.path.basename(fpath)

        # check if name is on the guest list
        if name in restr_names:
            fpaths_ret.append(fpath)

    return fpaths_ret


def safe_drop_suffix(s, suffix):
    """
    Drops a suffix if a string ends in the suffix.
    """
    if s.endswith(suffix):
        return s[:-len(suffix)]
    else:
        return s


def get_file_names(fpaths):
    """
    Gets the file names from a list of file paths without their extensions.

    Parameters
    ----------
    fpaths: list of str, str
        The file paths.

    Output
    ------
    fnames: list of str, str
        The file names without their extension.
    """
    if isinstance(fpaths, str):
        return pathlib.Path(fpaths).stem
    else:
        return [pathlib.Path(fpath).stem for fpath in fpaths]
