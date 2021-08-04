import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist, pdist, squareform
from Levenshtein import ratio
import pandas as pd
from multiprocessing import Process, Queue, Pool
import logging


def worker_pool_thresholds(input_list):
    data1 = input_list[0]
    data2 = input_list[1]
    k = input_list[2]
    new_distance_mat = cdist(data1, data2, lambda a, b: 1 - ratio(a[0], b[0]))
    new_distance_mat = np.sort(new_distance_mat, axis=-1)
    distance_mat_k = new_distance_mat[:, :k]
    return distance_mat_k

#
# def process_worker_detector(input_list):
#     queries = np.array(input_list[0])[:, None]
#     dists = cdist(queries, np.reshape(input_list[1], (-1, 1)), lambda a, b: 1 - ratio(a[0], b[0])).reshape(-1)
#     return dists


def get_nearest_queries(queries, query, k, pr: bool = False):
    """
    :param queries: DataFrame(text) or list of text
    :param query: one single string
    :param k: int for knn
    :param pr: to only print the results set True
    :return: k_index, k_nearest_dists, k_avg_dist
    """
    k = k

    queries = np.array(queries)[:, None]
    dists = cdist(queries, np.reshape(query, (-1, 1)),
                  lambda a, b: 1 - ratio(a[0], b[0])).reshape(-1)
    k_index = dists.argpartition(k-1, axis=0)[:k]
    k_nearest_dists = np.partition(dists, k - 1)[:k, None]
    # # transformed_strings
    # print(k_nearest_dists)
    k_avg_dist = np.mean(k_nearest_dists)
    # print(k_i)
    if pr:
        pr(k_index, "\n", k_nearest_dists, "\n", k_avg_dist)
    else:
        return k_index, k_nearest_dists, k_avg_dist


class Detector:
    """Create a new object of the class and call process() from the object"""

    def __init__(self, k: int = 25, threshold: float = 15):
        """
        k is the number of nearest neighbors of each new query to be considered
        threshold is a number at which an attack is detected if the distance of knn is higher.
        """
        self.K = k
        self.threshold = threshold
        self.num_queries = 0
        self.buffer = []
        self.list_of_buffers = []
        self.chunk_size = 100
        self.list_of_processes = []
        self.length_of_input = 0
        self.history_attack = []
        self.history = []  # detected attacks
        self.detected_dists = []  # detected knn distances
        self.str_distance = self.str_distance_ratio
        self.pdist_values = None

    @staticmethod
    def str_distance_ratio(a, b):
        return 1 - ratio(a, b)

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

    def once_process(self, queries, reset: bool = False):
        """calculate pdist once for the whole data and find knn for each query
        using right indexes of the pdist output.
        Just for large data without attack (large and benign data) is faster,
        otherwise process method is much faster"""
        if reset:
            self.__init__(self.K, self.threshold)
        self.__check_type(queries)
        k = self.K
        previous_attack_index = 0
        # if metric == "ratio":
        #     self.str_distance = self.str_distance_ratio
        self.length_of_input = len(queries)
        queries = np.array(queries)[:, None]
        query_index_reset = 0
        self.pdist_values = squareform(pdist(queries, lambda a, b: self.str_distance(a[0], b[0])))
        for query_index in tqdm(range(len(queries))):
            query_index_reset += 1
            if query_index_reset <= self.K:
                self.num_queries += 1
                continue
            dists = self.pdist_values[query_index, previous_attack_index:query_index]
            k_nearest_dists = np.partition(dists, k - 1)[:k, None]
            k_avg_dist = np.mean(k_nearest_dists)

            self.num_queries += 1
            is_attack = k_avg_dist < self.threshold
            if is_attack:
                previous_attack_index = query_index
                self.history.append(self.num_queries)
                query_index_reset = 0
                self.detected_dists.append(k_avg_dist)
                self.clear_memory()

    def process(self, queries, reset: bool = False):
        """

        :param queries: list of text or a pandas column of text
        :param reset: True for reseting the results in the class
        :return:
        """
        if reset:
            self.__init__(self.K, self.threshold)
        self.__check_type(queries)
        # if metric == "ratio":
        #     self.str_distance = self.str_distance_ratio
        self.length_of_input = len(queries)
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

    def multi_process(self, queries, chunk: int = 75, reset: bool = False, sless=0): #, pooling_process=0):
        if not isinstance(chunk, int) or not isinstance(reset, bool):
            raise ValueError(f"reset should be bool and chunk should be int!")
        if reset:
            self.__init__(self.K, self.threshold)
        # if metric == "ratio":
        #     self.str_distance = self.str_distance_ratio
        self.chunk_size = chunk
        self.__check_type(queries)
        self.length_of_input = len(queries)
        # if not pooling_process:
        for query in tqdm(queries):
            self.multi_process_query(query, sless)
        # elif pooling_process:
        #     for query in tqdm(queries):
        #         self.multi_process_query_pooling(query, pooling_process)
        # else:
        #     exit(1)

    # def multi_process_query_pooling(self, query, num_process):
    #     if len(self.buffer) < self.K:
    #         self.buffer.append(query)
    #         self.num_queries += 1
    #         return False
    #
    #     k = self.K
    #     input_lists = []
    #     px = len(self.buffer) // num_process if len(self.buffer) > num_process else 2
    #     px = 75
    #     for i in range(0, len(self.buffer), px):
    #         input_lists.append([self.buffer[i:i + px], query])
    #
    #     with Pool(num_process) as p:
    #         distances = p.map(process_worker_detector, input_lists)
    #
    #     dists = np.concatenate(distances)
    #     k_nearest_dists = np.partition(dists, k - 1)[:k, None]
    #     k_avg_dist = np.mean(k_nearest_dists)
    #
    #     self.buffer.append(query)
    #     self.num_queries += 1
    #
    #     is_attack = k_avg_dist < self.threshold
    #     if is_attack:
    #         self.history.append(self.num_queries)
    #         self.detected_dists.append(k_avg_dist)
    #         self.clear_memory()

    def multi_process_query(self, query, sless=0):
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

    def print_result(self, show_avg=False):
        print("\n")
        epochs = self.get_detections()
        num_attacks = len(epochs)
        if num_attacks == 0:
            print("\n\033[32m\t\t---->>No attack is detected!<<----\033[00m")
            print("\033[34m#Num of detections:\t", num_attacks, "\033[00m")
        else:
            print("\033[31m\t\t--->>>ATTACK DETECTED!<<<---\033[00m")
            print("\033[38;5;105mNum detections:\t", num_attacks, "\033[00m")

        print("Queries per detection:\t", epochs)
        print("i-th query that caused detection:\t", self.history)
        print("\033[36m#Num of queries\t>>>>>>>>\t", self.length_of_input, "\033[00m")
        if show_avg:
            print("Avg. distance Detected :\t", self.detected_dists,
                  "\n\t\t---------------END---------------")


class Thresholds:
    def __init__(self):
        self.list_of_k = []
        self.list_of_thresholds = []
        self.saved_history = []
        self.str_distance = self.str_distance_ratio

    @staticmethod
    def str_distance_ratio(a, b):
        return 1 - ratio(a, b)

    def calculate_thresholds(self, data, k, chunk: int = 100, up_to_k: bool = False,
                             multiprocess="singleprocess", num_process: int = 4, percentile=0.1, pair="cdist"):

        logging.info(f"calculate_thresholds(data, k={k}, chunk={chunk}, up_to_k={up_to_k}, multiprocess={multiprocess},"
                     f" num_process= {num_process}, percentile={percentile}, pair={pair})")
        logging.info(f"Input data length: {len(data)}")
        # if metric == "ratio":
        #     self.str_distance = self.str_distance_ratio
        data = np.reshape(data.values, (-1, 1))
        distances = []

        if multiprocess == "singleprocess":
            if pair == "pdist":
                print('Single processing - pdist - without loading - please wait...')
                new_distance_mat = pdist(data, lambda a, b: self.str_distance(a[0], b[0]))
                new_distance_mat = np.sort(squareform(new_distance_mat), axis=-1)
                # I think it's better to use new_distance_mat[:, 1:k+1] but I need to test it before changing
                distance_mat_k = new_distance_mat[:, :k]

                distances.append(distance_mat_k)
            else:
                print('Single processing - cdist')
                for i in tqdm(range(0, len(data), chunk)):
                    new_distance_mat = cdist(data[i:i + chunk, :], data, lambda a, b: self.str_distance(a[0], b[0]))
                    new_distance_mat = np.sort(new_distance_mat, axis=-1)
                    distance_mat_k = new_distance_mat[:, :k]

                    distances.append(distance_mat_k)

        elif multiprocess == "multiprocess":
            queue = Queue()
            chunk_changed = len(data) % chunk < num_process and len(data) % chunk != 0
            while len(data) % chunk < num_process and len(data) % chunk != 0:
                chunk += 1
            if chunk_changed:
                print(f"Warning: chunk size has changed to {chunk}")
            parts = len(data) // chunk if len(data) % chunk == 0 else (len(data) // chunk) + 1
            logging.info(f"{parts} Parts \t"
                         f"~{num_process} Processes for each part")
            print(
                f"\n\033[01;33m{parts} Parts...\n"
                f"~{num_process} Processes for each part\n\033[01;36mWait...\n\033[00m")

            for i in range(0, len(data), chunk):
                list_of_processes = []
                new_data = data[i:i + chunk, :]
                px = len(new_data) // num_process
                for inx in range(0, len(new_data), px):
                    tmp = new_data[inx:inx + px, :]
                    # self.saved_history.append([tmp, inx, inx + px, px])

                    prc = Process(target=self.__process_worker_thresholds, args=(tmp, data, k, queue,))
                    prc.start()
                    list_of_processes.append(prc)

                for each_pr in tqdm(list_of_processes):
                    each_pr.join()
                    distances.append(queue.get())

                # while not queue.empty():
                #     distances.append(queue.get())

                logging.info(f"queue getting, len(distances) : {len(distances)}")
        elif multiprocess == "pooling":
            print(f"Pool: {num_process}")
            input_lists = []

            for i in tqdm(range(0, len(data), chunk)):
                input_lists.append([data[i:i + chunk, :], data, k])

            with Pool(num_process) as p:
                distances = p.map(worker_pool_thresholds, input_lists)
        else:
            print(f"multiprocess argument should be defined properly. {multiprocess} is not defined!")
            exit(1)

        print("\033[38;5;105mWaiting for merging outputs.\033[00m")
        distance_matrix = np.concatenate(distances, axis=0)
        logging.info(f"Matrix length >>>>>>>> {len(distance_matrix)}")
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
