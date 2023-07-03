import numpy as np


class Compute_metrics():
    def __init__(self, connectivity_matrix, A, n_past) -> None:
        self.conn_mat = connectivity_matrix
        self.ground_truth = A
        self.number_of_past = n_past


    def compute_confusion_matrix(self, A):
        """

        :param inferred: Connectivity matrix inferred from data
        :param A: Ground truth connectivity matrix
        :return:
        """
        A, inferred = self.ground_truth, self.connectivity_matrix
        TP_inf = np.sum(np.logical_and(A != 0,inferred!=0))
        FN_inf = np.sum(np.logical_and(A != 0,inferred==0))
        FP_inf = np.sum(np.logical_and(A == 0,inferred!=0))
        TN_inf = np.sum(np.logical_and(A == 0,inferred==0))
        self.confusion_matrix = np.array([[TP_inf,FN_inf],
                                         [FP_inf,TN_inf]])
        return self.confusion_matrix


    def compute_metrics_from_confusion_matrix(self):
        """

        :param confusion_matrix:
        :return:
        """

        confusion_matrix_ = self.confusion_matrix.flatten()
        accuracy = (confusion_matrix_[0] + confusion_matrix_[3])/(np.sum(confusion_matrix_))
        precision = confusion_matrix_[0]/(confusion_matrix_[0]+confusion_matrix_[2])
        recall = confusion_matrix_[0]/(confusion_matrix_[0]+confusion_matrix_[1])
        FPR = confusion_matrix_[2]/(confusion_matrix_[2]+confusion_matrix_[3])
        return np.array([accuracy, precision, recall, FPR])


    def repopulate(inf,traces,idx):
        """

        :param inf:
        :param traces:
        :param idx:
        :return:
        """
        inferred_ = np.zeros((traces.shape[0], traces.shape[0]))
        p = np.transpose(np.where(inf != 0))
        for i in range(len(p)):
            inferred_[idx[p[i, 0]], idx[p[i, 1]]] = 1
        return inferred_


    def compute_distance_of_each_projection(self):
        """

        :param centers:
        :param inf:
        :return:
        """

        loc = np.transpose(np.where(inf>0))
        dist = np.zeros(len(loc))
        for i in range(len(loc)):
            p1,p2 = self.centers[loc[i,0]], self.centers[loc[i,1]] 
            dist[i] = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)
        return dist
        

    def count_number_of_in_out_edges_for_each_ROIs(out_,from_,emitter,reciever):
        """

        :param out_:
        :param from_:
        :param emitter:
        :param reciever:
        :return:
        """
        n_out,n_in = len(out_),len(from_)
        for el in out_:
            if el in emitter:
                n_out -= 1
        for el in from_:
            if el in reciever:
                n_in -= 1
        return n_out, n_in


    def roi_neighbors(self):
        """

        :param inferred_:
        :return:
        """
        inferred_ = self.conn_mat
        nodes_out, nodes_in = {},{}
        for i in range(inferred_.shape[0]):
            nodes_out[i] = np.where(inferred_[i]!=0)[0]
            nodes_in[i] = np.where(inferred_.T[i]!=0)[0]
        return nodes_out,nodes_in


    


# def single_res(input1,input2):
#     return input1 - (input1.transpose() @ input2) / (input2.transpose() @ input2) * input2
