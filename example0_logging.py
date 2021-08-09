from textdetection.detection import Detector
import pandas as pd
import time
import logging

logging.basicConfig(filename='outputs/mylogs/mylogs.log', format='%(asctime)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)


class datasetCSV():
    def __init__(self, path=None, text_column=None):
        self.path = path
        if self.path is None:
            raise ValueError(f"The path argument cannot be None! It should be string!")
        self.text_column = text_column
        if self.text_column is None:
            self.df = pd.read_csv(data_path)
        else:
            self.df = pd.read_csv(data_path)[self.text_column]


def detect_process_log(datasetObject, k=10, threshold=0.2, multiprocess=False, chunk=20):
    detector = Detector(k=k, threshold=threshold)
    start_time = time.time()
    if multiprocess:
        detector.multi_process(datasetObject.df, chunk=chunk, sless=0)
    elif not multiprocess:
        detector.process(datasetObject.df)
    elapsed_time = time.time() - start_time
    logging.info(
        f"time\t>>>>\t{elapsed_time} s \tnum of detections: {len(detector.history)}\n\tdata: {datasetObject.path}\t"
        f"len(data)={len(datasetObject.df)}\n\tDetector(k={k}, theshold={threshold}")
    detector.print_result()
    print(f"time elapsed : : : {elapsed_time}")
    return len(detector.history)


logging.info(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< NEW START  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

# data_path = "../data/suspicious/albert-base-v2-sst2_textbugger_sequences_2020-11-20-03-20.csv"  # text
# data_path = "../data/benign/glue.csv"  # sentence
# data_path = "../data/suspicious/cnn-ag-news_deepwordbug_sequences_2020-11-30-18-29.csv"  # text
# data_path = f"../data/benign/yelp_review_full_csv/new_versions/3_yelp.csv"  # text
# data_path = "../data/suspicious/bert-base-uncased-ag-news_deepwordbug_sequences_2020-11-30-17-40.csv"  # text
# data_path = "../data/suspicious/roberta-base-imdb_textbugger_sequences_2021-01-25-16-46.csv"  # text
# data_path = "../data/suspicious/lstm-yelp_textbugger_sequences_2021-01-25-15-58.csv"  # text
# data_path = "../data/suspicious/lstm-yelp_deepwordbug_sequences_2020-11-29-10-52.csv"  # text
# data_path = "../data/suspicious/cnn-yelp_textbugger_sequences_2020-11-29-11-19.csv"  # text
data_path = "../data/suspicious/cnn-yelp_deepwordbug_sequences_2020-11-20-15-34.csv"  # text
datasetObject = datasetCSV(path=data_path, text_column='text')
start_time = time.time()
num_of_detection = detect_process_log(datasetObject, k=10, threshold=0.21,
                                      multiprocess=False, chunk=60)
elapsed_time = time.time() - start_time
logging.info(f"<<<<<< FINISHED -- -- -- time passed: {elapsed_time} >>>>>>")
