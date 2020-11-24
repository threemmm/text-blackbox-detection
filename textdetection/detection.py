import numpy as np
from fuzzywuzzy import fuzz
from tqdm import tqdm
from scipy.spatial.distance import cdist


class Detector:

    def __init__(self, k=30, threshold=None):
        self.K = k
        self.threshold = threshold
        if self.threshold is None:
            raise ValueError("Must provide explicit textdetection threshold! For sentences shorter than 20 words, "
                             "10 is a good one")

        self.num_queries = 0
        self.buffer = []
        self.history = []  # detected attacks
        self.history_by_attack = []
        self.detected_dists = []  # detected knn distances

    @staticmethod
    def str_distance(a, b):
        return 100 - fuzz.partial_ratio(a, b)

    def process(self, queries):
        for query in tqdm(queries):
            self.process_query(query)

    def process_query(self, query):
        if len(self.buffer) < self.K:
            self.buffer.append(query)
            self.num_queries += 1
            return False

        k = self.K

        if len(self.buffer) > 0:
            queries = np.array(self.buffer)[:, None]
            dists = np.concatenate(cdist(queries, np.reshape(query, (-1, 1)), self.str_distance))

        k_nearest_dists = np.partition(dists, k - 1)[:k, None]
        k_avg_dist = np.mean(k_nearest_dists)

        self.buffer.append(query)
        self.num_queries += 1

        is_attack = k_avg_dist < self.threshold
        if is_attack:
            self.history.append(self.num_queries)
            self.detected_dists.append(k_avg_dist)
            self.clear_memory()

    def clear_memory(self):
        self.buffer = []

    def get_detections(self):
        history = self.history
        epochs = [history[0]]
        for i in range(len(history) - 1):
            epochs.append(history[i + 1] - history[i])

        return epochs
