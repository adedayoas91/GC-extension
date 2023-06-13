import mat73
import numpy as np
from pathlib import Path
from matplotlib.pyplot import plt


class load_data() -> ndArray:   # find the problem here

    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.file_type = self.file_path.stem

    
    def load(self): 
        """

        :param path:
        :return:
        """
        if self.type == 'npy':
            data_array = np.load(self.file_path)
        elif self.type == 'txt':
            data_array = np.loadtxt(self.file_path)
        else:
            raise NotImplementedError
        return data_array 





# class Database:

#     def __init__(self, data_set_no=2):
#         data_dict = mat73.loadmat('NoStim_Data.mat')
#         data  = data_dict['NoStim_Data']

#         deltaFOverF_bc = data['deltaFOverF_bc'][data_set_no]
#         derivatives = data['derivs'][data_set_no]
#         NeuronNames = data['NeuronNames'][data_set_no]
#         fps = data['fps'][data_set_no]
#         States = data['States'][data_set_no]


#         self.states = np.sum([n*States[s] for n, s in enumerate(States)], axis = 0).astype(int) # making a single states array in which each number corresponds to a behaviour
#         self.state_names = [*States.keys()]
#         self.neuron_traces = np.array(deltaFOverF_bc).T
#         self.derivative_traces = derivatives['traces'].T
#         self.neuron_names = np.array(NeuronNames, dtype=object)
#         self.fps = fps

#         f = open('readme.txt', 'r')
#         self.DESCR = f.read()
#         f.close()
#         '''
#         #Sort the data according to the clustering dendogram (only for dataset 3, as of now)
#         self.neuron_traces = self.neuron_traces[sort_indices]
#         self.derivative_traces = self.derivative_traces[sort_indices]
#         self.NeuronNames = self.NeuronNames[sort_indices]
#         '''
#         ## Creating dictionary of identified neurons and their indices
#         #self.neuron_id = {}
#         #for n, i in enumerate(self.NeuronNames):
#         #    if type(i) == list:
#         #        self.neuron_id[i[0]]=n

# def plot_raster(neuron_traces, derivative_traces):
#     fig, ax = plt.subplots(2,1, figsize=(15,10))
#     plt0 = ax[0].imshow(neuron_traces, aspect="auto", vmin=0, vmax=1)
#     #ax[0].set_yticks(np.arange(neuron_names.shape[0]))
#     #ax[0].set_yticklabels(neuron_names)
#     fig.colorbar(plt0, ax=ax[0])
#     plt1 = ax[1].imshow(derivative_traces, cmap='seismic', aspect="auto", vmin=-0.25, vmax=0.25)
#     #ax[1].set_yticks(np.arange(neuron_names.shape[0]))
#     #ax[1].set_yticklabels(neuron_names)
#     fig.colorbar(plt1, ax=ax[1])
#     plt.show()



# def dendogram(classifier):
#     ## Dendogram
#     # Create linkage matrix and then plot the dendrogram
#     # create the counts of samples under each node
#     counts = np.zeros(classifier.children_.shape[0])
#     n_samples = len(classifier.labels_)
#     for i, merge in enumerate(classifier.children_):
#         current_count = 0
#         for child_idx in merge:
#             if child_idx < n_samples:
#                 current_count += 1  # leaf node
#             else:
#                 current_count += counts[child_idx - n_samples]
#         counts[i] = current_count
#     linkage_matrix = np.column_stack([classifier.children_, classifier.distances_, counts]).astype(float)
#     # Plot the corresponding dendrogram
#     R = dendrogram(linkage_matrix, truncate_mode='level')
#     ## Sorting: The features are ordered according to the order of the leaves in the dendogram
#     sort_indices = R['leaves']
#     return sort_indices
