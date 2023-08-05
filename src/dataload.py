import numpy as np
from pathlib import Path


class LoadData:
    """

    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.file_type = self.file_path.suffix

    def load(self): 
        """

        Returns:
        """
        if self.file_type == 'npy':
            data_array = np.load(str(self.file_path)) # TODO: test if this works
        elif self.file_type == 'txt':
            data_array = np.loadtxt(self.file_path)
        else:
            raise NotImplementedError
        return data_array
