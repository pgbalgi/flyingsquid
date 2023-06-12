from sklearn.cluster import KMeans
import numpy as np

from flyingsquid.label_model import LabelModel

class EmbeddingModel:
    def __init__(self, n_clusters, class_balance):
        self.cluster_model = KMeans(n_clusters)
        self.label_models = [LabelModel(class_balance)] * n_clusters

    
    def fit(self, embeddings, annotations):
        n_clusters = self.cluster_model.get_params(deep=False)['n_clusters']
        clusters = self.cluster_model.fit_predict(embeddings)

        for i in range(n_clusters):
            cluster = np.nonzero(clusters == i)
            self.label_models[i].fit(annotations[cluster])
    

    def predict(self, embeddings, annotations):
        n_clusters = self.cluster_model.get_params(deep=False)['n_clusters']
        clusters = self.cluster_model.predict(embeddings)

        n = embeddings.shape[:-1]
        d = annotations.shape[-1]
        pred = np.zeros(n + (d,))

        for i in range(n_clusters):
            cluster = np.nonzero(clusters == i)
            pred[cluster] = self.label_models[i].predict(annotations[cluster])

        return pred