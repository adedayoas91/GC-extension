import mat73
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class Load_data():   # find the problem here

    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.file_type = self.file_path.suffix

    
    def load(self): 
        """

        :param path:
        :return:
        """
        if self.file_type == 'npy':
            data_array = np.load(self.file_path)
        elif self.file_type == 'txt':
            data_array = np.loadtxt(self.file_path)
        else:
            raise NotImplementedError
        return data_array 