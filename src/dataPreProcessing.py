import numpy as np

class PreProcess_data():
    def __init__(self, arr:np.ndarray, n_past:int=0, bad_frames:np.ndarray=None):
        self.orig_arr=arr.copy()
        self.n_past=n_past
        self.arr = self.sort_ROI_to_rows()
        self.bad_frames = bad_frames

    def sort_ROI_to_rows(self):
        i, j = self.orig_arr.shape()
        if i > j:
            self.arr = self.orig_arr.T
    
    def replace_nan_(self, bad_frames:int, delete_frames:bool):
        """
        Deletes NAN frames in data and if delete_frames is False, it interpolates to fill the bad frames
        Args:
            data: (array-like: matrix [# of variables X # of samples])
            bad_frames: (array-like: vector) Index of identified defected frames
            delete_frames: (bool) If true, deletes frames and if False, interpolates

        Returns: Data without bad frames
        """
        if delete_frames:
            self.arr = np.delete(self.arr, np.array(bad_frames), axis=1)
        else:
            for a in range(self.arr.shape[0]):
                for i in bad_frames:
                    self.arr[a,i] = (self.arr[a,i-1]+self.arr[a,i+1])/2
        return

    
    def make_shifted_versions_of_data(self):
        """
        Creates shifted versions of original data based on the number of pasts nn required.
        Concatenate the shifted arrays and return the new data.
        Args:
            X: (array-like, shape [# of variable X # of samples]): Raw data
            nn: (int) number of shifted version of data required

        Returns: shifted data

        """
        
        if self.n_past == 0:
            return self.arr
        else:
            X_ = self.arr[:, (self.n_past):]
            for i in range(self.n_past):
                idx1, idx2 = self.n_past-1-i, -i-1
                X_ = np.r_[X_, self.arr[:, idx1:idx2]]
            return X_
