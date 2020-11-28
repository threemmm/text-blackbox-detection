import numpy as np
from fuzzywuzzy import fuzz
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pandas as pd
from multiprocessing import Process, Queue
from time import sleep


class Detector:
    """Create a new object of the class and call process() from the object"""

    def __init__(self, k: int = 25, threshold: int = 15):
        """
        k is the number of nearest neighbors of each query that are accounted for detecting an attack
        threshold is a number between 0 to 100 and shows that if the average of distances of k neighbors
        are less than this value, an attack is detected.
        """
        self.K = k
        self.threshold = threshold
        self.num_queries = 0
        self.buffer = []
        self.list_of_buffers = []
        self.chunk_size = 100
        self.list_of_processes = []
        self.history_attack = []
        self.history = []  # detected attacks
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
                f"Input cannot be a string, it is expected to be a list of strings or"
                f" one column (DataFrame) of strings")

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
        dists = cdist(queries, np.reshape(query, (-1, 1)), lambda a, b: self.str_distance(a[0], b[0])).reshape(-1)
        k_nearest_dists = np.partition(dists, k - 1)[:k, None]
        k_avg_dist = np.mean(k_nearest_dists)

        self.buffer.append(query)
        self.num_queries += 1
        is_attack = k_avg_dist < self.threshold
        if is_attack:
            self.history.append(self.num_queries)
            self.detected_dists.append(k_avg_dist)
            self.clear_memory()

    def multi_process(self, queries, chunk: int = 75, reset: bool = False, sless: int = 0):
        if not isinstance(chunk, int) or not isinstance(reset, bool):
            raise ValueError(f"reset should be bool and chunk should be int!")
        if reset:
            self.__init__(self.K, self.threshold)
        self.chunk_size = chunk
        self.__check_type(queries)
        for query in tqdm(queries):
            self.multi_process_query(query, sless)

    def multi_process_query(self, query, sless):
        if not self.list_of_buffers and len(self.buffer) < self.K:
            self.buffer.append(query)
            self.num_queries += 1
            return False

        k = self.K
        all_dists = []
        queue = Queue()

        if self.buffer:
            buffer_process = Process(target=self.__process_worker, args=(self.buffer, query, queue))
            buffer_process.start()

        if self.list_of_buffers:
            for each_buffer in self.list_of_buffers:
                p = Process(target=self.__process_worker, args=(each_buffer, query, queue))
                p.start()
                self.list_of_processes.append(p)

            for each_pr in self.list_of_processes:
                each_pr.join()

            self.list_of_processes = []

        if self.buffer:
            buffer_process.join()
        while not queue.empty():
            all_dists.append(queue.get())

        dists = np.concatenate(all_dists)
        k_nearest_dists = np.partition(dists, k - 1)[:k, None]
        k_avg_dist = np.mean(k_nearest_dists)

        self.buffer.append(query)
        self.num_queries += 1

        if len(self.buffer) >= self.chunk_size:
            self.list_of_buffers.append(self.buffer)
            self.buffer = []

        if k_avg_dist < sless:
            self.history_attack.append([self.num_queries, query, k_nearest_dists, k_avg_dist])
        is_attack = k_avg_dist < self.threshold
        if is_attack:
            self.history.append(self.num_queries)
            self.detected_dists.append(k_avg_dist)
            self.clear_memory()

    def __process_worker(self, input_each_buffer, query, queue):
        queries = np.array(input_each_buffer)[:, None]
        dists = cdist(queries, np.reshape(query, (-1, 1)), lambda a, b: self.str_distance(a[0], b[0])).reshape(-1)
        queue.put(dists)

    def clear_memory(self):
        self.buffer = []
        self.list_of_buffers = []

    def get_detections(self):
        history = self.history
        epochs = []
        if not history:
            return epochs
        else:
            epochs = [history[0]]
        for i in range(len(history) - 1):
            epochs.append(history[i + 1] - history[i])

        return epochs

    def print_result(self):
        print("\n")
        epochs = self.get_detections()
        num_attacks = len(epochs)
        if num_attacks == 0:
            print("\n\033[32m---No attack is detected!---\033[00m")
            print("\033[38;5;105mNum detections:", num_attacks, "\033[00m")
        else:
            print("\033[31m--->>>ATTACK DETECTED!<<<---\033[00m")
            print("\033[38;5;105mNum detections:", num_attacks, "\033[00m")

        print("Queries per detection:", epochs)
        print("i-th query that caused detection:", self.history)


class Thresholds:
    def __init__(self):
        self.list_of_k = []
        self.list_of_thresholds = []
        self.saved_history = []

    @staticmethod
    def str_distance(a, b):
        return 100 - fuzz.partial_ratio(a, b)

    # def chunks(lst, n):
    #     """Yield successive n-sized chunks from lst."""
    #     for i in range(0, len(lst), n):
    #         yield lst[i:i + n]

    def calculate_thresholds(self, data, k, chunk: int = 100, up_to_k: bool = False,
                             multiprocess=False, num_process: int = 4, percentile = 0.1):
        data = np.reshape(data.values, (-1, 1))
        distances = []
        process_list = []

        if not multiprocess:
            print('single processing')
            for i in tqdm(range(0, len(data), chunk)):
                new_distance_mat = cdist(data[i:i + chunk, :], data, lambda a, b: Detector.str_distance(a[0], b[0]))
                new_distance_mat = np.sort(new_distance_mat, axis=-1)
                distance_mat_k = new_distance_mat[:, :k]

                distances.append(distance_mat_k)

        elif multiprocess:
            queue = Queue()
            print(
                f"\n\033[01;33m{len(data) // chunk + 1} Parts...\n~{num_process} Processes for each part\n\033[01;36mWait...\n\033[00m")

            for i in range(0, len(data), chunk):
                list_of_processes = []
                new_data = data[i:i + chunk, :]
                px = len(new_data) // num_process
                for inx in range(0, len(new_data), px):
                    tmp = new_data[inx:inx + px, :]
                    self.saved_history.append([tmp, inx, inx + px, px])

                    prc = Process(target=self.__process_worker_thresholds, args=(tmp, data, k, queue,))
                    prc.start()
                    list_of_processes.append(prc)

                for each_pr in tqdm(list_of_processes, leave=True):
                    each_pr.join()
                while not queue.empty():
                    distances.append(queue.get())

            print("\n\033[38;5;105mWaiting for merging outputs.\033[00m")

        else:
            # todo
            print("multiprocess= argument should set bool, e.g. multiprocess=True")
            exit(1)
            queue = Queue()
            for i in tqdm(range(0, len(data), chunk)):
                tmp = data[i:i + chunk, :]
                prc = Process(target=self.__process_worker_thresholds, args=(tmp, data, k, queue,))
                # sleep(1)
                prc.start()
                process_list.append(prc)

                # sleep()
            # print("\033[38;5;105m", len(list_of_processes), " Processe(s) is/are created.\033[00m\n-----")

            print("\033[38;5;105mWaiting for merging outputs.\033[00m")
            sleep(2)
            # for pp in process_list:
            #     if pp.is_alive():
            #         pp.join()
            #     else:
            #         continue
            process_list[0].join()
            while not queue.empty():
                distances.append(queue.get())

        distance_matrix = np.concatenate(distances, axis=0)
        print("Matrix length >>>>>>>> ", len(distance_matrix))

        def __last_assigning():
            dist_to_k_neighbors = distance_matrix[:, :k + 1]
            avg_dist_to_k_neighbors = dist_to_k_neighbors.mean(axis=-1)
            threshold = np.percentile(avg_dist_to_k_neighbors, percentile)
            self.list_of_k.append(k)
            self.list_of_thresholds.append(threshold)

        if up_to_k:
            for k in tqdm(range(0, k + 1)):
                __last_assigning()
        else:
            __last_assigning()

        return k, self.list_of_thresholds[-1]

    def __process_worker_thresholds(self, data1, data2, k, queue):
        new_distance_mat = cdist(data1, data2, lambda a, b: self.str_distance(a[0], b[0]))
        new_distance_mat = np.sort(new_distance_mat, axis=-1)
        distance_mat_k = new_distance_mat[:, :k]
        queue.put(distance_mat_k)
