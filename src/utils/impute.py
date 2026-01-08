from sklearn.impute import KNNImputer

class AirnetImputer(KNNImputer):
    def __init__(self,numNeighbors):
        super(AirnetImputer,self).__init__(n_neighbors=numNeighbors)
        super().set_output(transform='pandas')
        self.numNeighbors = numNeighbors
