import numpy as np

class PreProcess_data():
    def __init__(self, arr, n_past, bad_frames=bool):
        self.original_data = arr.copy()
        self.orig_arr_sorted = self.sort_ROI_to_rows()
        self.arr = None

    def sort_ROI_to_rows(self):
        i, j = self.original_data.shape()
        if i > j:
            self.arr = self.original_data.T
    
    def replace_nan_(self, bad_frames, delete_frames = bool):
        """
        Deletes NAN frames in data and if delete_frames is False, it interpolates to fill the bad frames
        Args:
            data: (array-like: matrix [# of variables X # of samples])
            bad_frames: (array-like: vector) Index of identified defected frames
            delete_frames: (bool) If true, deletes frames and if False, interpolates

        Returns: Data without bad frames
        """
        data = self.arr
        if delete_frames:
            data = np.delete(data, np.array(bad_frames), axis=1)
        else:
            for a in range(data.shape[0]):
                for i in bad_frames:
                    data[a,i] = (data[a,i-1]+data[a,i+1])/2
        return data
