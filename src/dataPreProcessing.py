import numpy as np

def replace_nan_(data, frames, delete_frames = bool):
    """
    Deletes NAN frames in data
    Args:
        data: (array-like: matrix [# of variables X # of samples])
        frames: (array-like: vector) Index of identified defected frames
        delete_frames:

    Returns:

    """
    if delete_frames:
        data = np.delete(data,np.array(frames),axis=1)
    else:
        for a in range(data.shape[0]):
            for i in frames:
                data[a,i] = (data[a,i-1]+data[a,i+1])/2
    return data
