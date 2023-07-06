# import mat73
import numpy as np
from pathlib import Path
# import matplotlib.pyplot as plt


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




# #############
# def load_pickle_data(self):
#     raise NotImplementedError
#
# def get_number_of_perm(self):
#     return self.number_of_perm
#
# def dependence_test_method(self, dependence_test_choice):
#     self.dependence_test = dependence_test_choice
#
# def get_data_path(self, file_path):
#     return file_path
#
# def get_bad_frames(self, bad_frame_times):
#     return bad_frame_times
#
# def load_data(self, file_path):
#     """
#     File name should be given in string with the path to where it is located.
#     :param file_name: the name of file containing data.
#     :return: traces (loaded data)
#     """
#     self.traces = np.load(file_path)
#     return self.traces
#
