import numpy as np


class Compute_metrics():
    def __init__(self, connectivity_matrix, ) -> None:
        pass


    def confusion_matrix(inferred, A):
        """

        :param inferred:
        :param A:
        :return:
        """
        TP_inf = np.sum(np.logical_and(A != 0,inferred!=0))
        FN_inf = np.sum(np.logical_and(A != 0,inferred==0))
        FP_inf = np.sum(np.logical_and(A == 0,inferred!=0))
        TN_inf = np.sum(np.logical_and(A == 0,inferred==0))
        return np.array([[TP_inf,FN_inf],
                        [FP_inf,TN_inf]])


    def apr_metrics(confusion_matrix):
        """

        :param confusion_matrix:
        :return:
        """
        confusion_matrix = confusion_matrix.flatten()
        accuracy = (confusion_matrix[0] + confusion_matrix[3])/(np.sum(confusion_matrix))
        precision = confusion_matrix[0]/(confusion_matrix[0]+confusion_matrix[2])
        recall = confusion_matrix[0]/(confusion_matrix[0]+confusion_matrix[1])
        FPR = confusion_matrix[2]/(confusion_matrix[2]+confusion_matrix[3])
        return np.array([accuracy, precision, recall, FPR])


    def repopulate(inf,traces,idx):
        """

        :param inf:
        :param traces:
        :param idx:
        :return:
        """
        inferred_ = np.zeros((traces.shape[0],traces.shape[0]))
        p = np.transpose(np.where(inf!=0))
        for i in range(len(p)):
            inferred_[idx[p[i,0]], idx[p[i,1]]] = 1
        return inferred_


    def distance(centers,inf):
        """

        :param centers:
        :param inf:
        :return:
        """
        loc = np.transpose(np.where(inf>0))
        dist = np.zeros(len(loc))
        for i in range(len(loc)):
            p1,p2 = centers[loc[i,0]],centers[loc[i,1]] 
            dist[i] = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)
        return dist
        

    def counter_(out_,from_,emitter,reciever):
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


    def vertixDegree(inferred_):
        """

        :param inferred_:
        :return:
        """
        nodes_out,nodes_in = {},{}
        for i in range(inferred_.shape[0]):
            nodes_out[i] = np.where(inferred_[i]!=0)[0]
            nodes_in[i] = np.where(inferred_.T[i]!=0)[0]
        return nodes_out,nodes_in


    


# def single_res(input1,input2):
#     return input1 - (input1.transpose() @ input2) / (input2.transpose() @ input2) * input2
