import numpy as np
from fuzzywuzzy import fuzz
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pandas as pd


class Detector:
    """Create a new object of the class and call process() from the object"""

    def __init__(self, k: int = 25, threshold: int = 15):
        """
        k is the number of nearest neighbors of each query that are accounted for detecting an attack
        threshold is a number between 0 to 100 and shows that if the average of distances of k neighbers
        are less than this value, an attack is detected.
        """
        self.K = k
        self.threshold = threshold
        self.num_queries = 0
        self.buffer = []
        self.history = []  # detected attacks
        self.history_by_attack = []
        self.detected_dists = []  # detected knn distances

    @staticmethod
    def str_distance(a, b):
        return 100 - fuzz.partial_ratio(a, b)

    @staticmethod
    def __check_type(data):
        if isinstance(data, list):
            if not data:
                raise ValueError(f"Input {type(data)} is empty!")
            elif not isinstance(data[0], str):
                raise ValueError(f"Input {type(data[0])} is not acceptable!")
        elif isinstance(data, pd.Series):
            if data.empty:
                raise ValueError(f"Input {type(data)} could not be empty!")
            elif not isinstance(data.iloc[0], str):
                raise ValueError(f"Input {type(data[0])} is not acceptable!")
        elif isinstance(data, pd.DataFrame):
            raise ValueError(
                f"Input {type(data)} is not acceptable! One column of Strings or list of Strings is expected!")
        elif isinstance(data, str):
            raise ValueError(
                f"Input cannot be a string, it is expected to be a list of strings or one column (DataFrame) of strings")

    def process(self, queries, reset: bool = False):
        if reset:
            self.__init__(self.K, self.threshold)
        self.__check_type(queries)
        for query in tqdm(queries):
            self.process_query(query)

    def process_query(self, query):
        if len(self.buffer) < self.K:
            self.buffer.append(query)
            self.num_queries += 1
            return False

        k = self.K

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
        epochs = []
        if not history:
            print("\nNo attack is detected!")
        else:
            epochs = [history[0]]
        for i in range(len(history) - 1):
            epochs.append(history[i + 1] - history[i])

        return epochs

    def print_result(self):
        detections = self.get_detections()
        print("Num detections:", len(detections))
        print("Queries per detection:", detections)
        print("i-th query that caused detection:", self.history)
