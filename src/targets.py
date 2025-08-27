'''
FROM https://github.com/shooash/targets
Path resolution helper to quickly generate absolute paths.
Usage:
Create a class with a decorator to hold paths relative to .py file location (anchor) as TargetPath

from targets import targets, TargetPath

@targets(__file__, ensure=True) # Script's .py file location, ensure that the directories exist
class DataDir(Targets):
    DATA = TargetPath('data')
    RAW = TargetPath('data/raw')
    PROCESSED = TargetPath('data/processed')

CLEAN_FILE = DataDir.RAW('clean.csv') # = ..../data/raw/clean.csv (absolute path)

os.makedirs(DataDir.PROCESSED) # creates .../data/processed
print(DataDir.DATA) # outputs '/data'
child_dir = DataDir.PROCESSED.split('/')[-1] # = 'processed' as TargetPath inherits from str

'''

import os

class TargetPath(str):
    def __init__(self, relative_path : str):
        self.relative_path = relative_path
        self.path = ''
        super().__init__()
    def _add_anchor(self, anchor : str):
        path = os.path.abspath(os.path.join(anchor, self.relative_path))
        new_target_path = TargetPath(path)
        new_target_path.path = path
        return new_target_path
    def ensure(self):
        os.makedirs(self.path, exist_ok=True)
    def __call__(self, filename : str | list[str] = '') -> str | list[str]:
        if isinstance(filename, str):
            return os.path.join(self.path, filename or '')
        if isinstance(filename, list):
            return [os.path.join(self.path, f or '') for f in filename]
    def __repr__(self):
        return self.path or self.relative_path
    def __str__(self):
        return self.path or self.relative_path
        
def targets(anchor : str, ensure = False):
    if not os.path.isdir(anchor):
        anchor = os.path.dirname(anchor)
    def updater(cls):
        for k, v in cls.__dict__.copy().items():
            if k.startswith('_'):
                continue
            if isinstance(v, TargetPath):
                new_v = v._add_anchor(anchor)
                setattr(cls, k, new_v)
                if ensure:
                    new_v.ensure()
        return cls
    return updater
