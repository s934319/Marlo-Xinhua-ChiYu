import os
import errno

def CreatFolder(folder_name):
    try:
        os.makedirs(folder_name)
    except FileExistsError:
        # directory already exists
        pass
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
