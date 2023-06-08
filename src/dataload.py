import mat73
import numpy as np



def load_data(path):
    traces = np.load('path')
    return traces 


def simulate_data(A, m, iid = bool, latency = bool):
    """
    Function to create a iid dataset for analysis

    Parameters
    A (matrix) - Adjacency matrix (A lower triangular matrix - topological order of the variables)
    m (int) - desired length of the variable
    iid (bool) - Specify data to be simulated.

    Returns
    data (matrix) with shape [A.shape[0],m]

    Example of A =  np.array([[0,0,0,0,0],
                              [1,0,0,0,0],
                              [0,1,0,0,0],
                              [1,1,0,0,0],
                              [0,0,1,0,0]])

    """
    np.random.seed(10)
    if iid==True:
        X = np.zeros([A.shape[0],m]).T
        for i, row in enumerate(X):
            for n, var in enumerate(row):
                X[i, n] = np.random.normal(0, 0.1) + np.dot(A[n], X[i])
    else:
        X = np.zeros([A.shape[0],m]).T
        X[0] = np.random.randn(A.shape[0])
        for i, row in enumerate(X[:-1]):
            if latency == False:
                X[i+1] = A @ X[i] + np.random.normal(0,0.25,A.shape[0])
            else: 
                X[i+1] = A @ X[i] + np.random.normal(0,0.25,A.shape[0]) + noise[:,i]

    return X.T


def continuous_noise_fun(num, l):
    xx = np.linspace(0,500,l)
    noise = np.zeros((num,l))
    for i in range(num):
        a = 2*np.random.normal(0,0.25,size=6)
        c = 500*(np.random.random(size=6))
        s = 1+100*(np.random.random(size=6))
        yy = 0*xx
        for j in range(6):
            yy = yy + a[j]*np.exp(-(xx-c[j])**2/s[j])
        noise[i] = yy
    return noise



class Database:

    def __init__(self, data_set_no=2):
        data_dict = mat73.loadmat('NoStim_Data.mat')
        data  = data_dict['NoStim_Data']

        deltaFOverF_bc = data['deltaFOverF_bc'][data_set_no]
        derivatives = data['derivs'][data_set_no]
        NeuronNames = data['NeuronNames'][data_set_no]
        fps = data['fps'][data_set_no]
        States = data['States'][data_set_no]


        self.states = np.sum([n*States[s] for n, s in enumerate(States)], axis = 0).astype(int) # making a single states array in which each number corresponds to a behaviour
        self.state_names = [*States.keys()]
        self.neuron_traces = np.array(deltaFOverF_bc).T
        self.derivative_traces = derivatives['traces'].T
        self.neuron_names = np.array(NeuronNames, dtype=object)
        self.fps = fps

        f = open('readme.txt', 'r')
        self.DESCR = f.read()
        f.close()
        '''
        #Sort the data according to the clustering dendogram (only for dataset 3, as of now)
        self.neuron_traces = self.neuron_traces[sort_indices]
        self.derivative_traces = self.derivative_traces[sort_indices]
        self.NeuronNames = self.NeuronNames[sort_indices]
        '''
        ## Creating dictionary of identified neurons and their indices
        #self.neuron_id = {}
        #for n, i in enumerate(self.NeuronNames):
        #    if type(i) == list:
        #        self.neuron_id[i[0]]=n

def plot_raster(neuron_traces, derivative_traces):
    fig, ax = plt.subplots(2,1, figsize=(15,10))
    plt0 = ax[0].imshow(neuron_traces, aspect="auto", vmin=0, vmax=1)
    #ax[0].set_yticks(np.arange(neuron_names.shape[0]))
    #ax[0].set_yticklabels(neuron_names)
    fig.colorbar(plt0, ax=ax[0])
    plt1 = ax[1].imshow(derivative_traces, cmap='seismic', aspect="auto", vmin=-0.25, vmax=0.25)
    #ax[1].set_yticks(np.arange(neuron_names.shape[0]))
    #ax[1].set_yticklabels(neuron_names)
    fig.colorbar(plt1, ax=ax[1])
    plt.show()



def dendogram(classifier):
    ## Dendogram
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(classifier.children_.shape[0])
    n_samples = len(classifier.labels_)
    for i, merge in enumerate(classifier.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([classifier.children_, classifier.distances_, counts]).astype(float)
    # Plot the corresponding dendrogram
    R = dendrogram(linkage_matrix, truncate_mode='level')
    ## Sorting: The features are ordered according to the order of the leaves in the dendogram
    sort_indices = R['leaves']
    return sort_indices
