from textdetection.detection import Detector, Thresholds
import pandas as pd
import time
import pickle
import logging
import os
from matplotlib import pyplot as plt

logging.basicConfig(filename='outputs/mylogs/mylogs.log', format='%(asctime)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)
logging.info(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< NEW START  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

data_path = "../data/suspicious/albert-base-v2-sst2_textbugger_sequences_2020-11-20-03-20.csv"  # text
data_path ="../data/benign/glue.csv" #sentence
# data_path = "../data/suspicious/cnn-ag-news_deepwordbug_sequences_2020-11-30-18-29.csv"  # text
# data_path = f"../data/benign/yelp_review_full_csv/new_versions/3_yelp.csv"  # text
data_path = "../data/suspicious/bert-base-uncased-ag-news_deepwordbug_sequences_2020-11-30-17-40.csv" #text
data_path = "../data/suspicious/roberta-base-imdb_textbugger_sequences_2021-01-25-16-46.csv" #text
data_path = "../data/suspicious/lstm-yelp_textbugger_sequences_2021-01-25-15-58.csv" #text
data_path = "../data/suspicious/lstm-yelp_deepwordbug_sequences_2020-11-29-10-52.csv" #text
data_path = "../data/suspicious/cnn-yelp_textbugger_sequences_2020-11-29-11-19.csv" #text
data_path = "../data/suspicious/cnn-yelp_deepwordbug_sequences_2020-11-20-15-34.csv"  # text
data_with_path = []


# data_with_path = [pd.read_csv(data_path)['text'], data_path]


# df = pd.read_csv(data_path)['text']

def detect_process_log(data_with_path, k=10, threshold=0.2, multiprocess=False, chunk=20):
    detector = Detector(k=k, threshold=threshold)
    # logging.info(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<  detect_process_log  >>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # logging.info(f"Detector(k={k}, theshold={threshold})")
    start_time = time.time()
    if multiprocess:
        detector.multi_process(data_with_path[0], chunk=chunk, sless=0)
    elif not multiprocess:
        detector.process(data_with_path[0])
        # detector.once_process(data_with_path[0]) #calculate pdist once (for large and benign data)
    elapsed_time = time.time() - start_time
    logging.info(
        f"time\t>>>>\t{elapsed_time} s \tnum of detections: {len(detector.history)}\n\tdata: {data_with_path[1]}\t"
        f"len(data)={len(data_with_path[0])}\n\tDetector(k={k}, theshold={threshold}")
    detector.print_result()
    print(f"time elapsed : : : {elapsed_time}")
    # logging.info(f"data: {data_with_path[1]}  len(data)={len(data_with_path[0])}  --- process(df,)")
    # epochs = detector.get_detections()
    # num_attacks = len(epochs)
    # logging.info(f"")
    # logging.info(f"Queries per detection: {epochs}")
    # logging.info(f"i-th query that caused detection: {detector.history}")
    # logging.info(f"Num of queries: {detector.length_of_input}")
    # logging.info(f"Avg. distance Detected:\t {detector.detected_dists}")
    # detector.print_result()
    return len(detector.history)


K = 15
THRESHOLD = 0.15
all_result = []
SAVE_PICKLE_PLOT = False

# for ix in range(1):
if True:
    data_version = num_of_detection_list = []
    file_name = f"cnn-ag-news_deepwordbug_sequences_2020-11-30-18-29"
    start_time = time.time()

    for i in range(1, 2):
        # file_name = f"yelp"
        # data_path = f"../data/benign/yelp_review_full_csv/new_versions/{i}_" + file_name + ".csv"  # text
        # data_path = f"../yelp_review_full_csv/new_versions/{i}_" + file_name + ".csv"#text
        # data_path = f"../data/benign/glue_series/{i}_" + file_name + ".csv"#text
        data_with_path = [pd.read_csv(data_path)['text'], data_path]
        num_of_detection = detect_process_log(data_with_path, k=K, threshold=THRESHOLD,
                                              multiprocess=False, chunk=60)
        print(f">>>>>>>>>>> {i} process finished.")
        data_version.append(i)
        num_of_detection_list.append(num_of_detection)
    elapsed_time = time.time() - start_time
    logging.info(f"<<<<<< time passed: {elapsed_time} >>>>>>")

    K += 1
    THRESHOLD += 0.01
    result = [file_name, K, THRESHOLD, data_version, num_of_detection_list]
    str_time = time.strftime("%m-%d_%H-%M")
    # if not os.path.exists(f"outputs/pickle_list"):
    #     os.makedirs(f"outputs/pickle_list")
    # file_name_pickle = f"outputs/pickle_list/pickle_{str_time}.txt"
    # with open(file_name_pickle, "wb") as fp:  # Pickling
    #     pickle.dump(result, fp)
    # logging.info(f"The list of results was saved into: {file_name_pickle}")

    if SAVE_PICKLE_PLOT:
        # str_time = time.strftime("%m-%d_%H-%M-%s")
        f, ax = plt.subplots(1)
        ax.plot(result[3], result[4])
        # ax.set_ylim(ymin=0)
        # ax.set_xlim(xmin=0)
        plt.xlabel(f"#x {file_name}")
        plt.ylabel("#Num of detection")
        # plt.plot(results_of_k, results_thresholds)
        # plt.show()

        file_name_plot_PDF = f"outputs/pickle_list/plot_k_{result[1]}_threshold_{'%.2f' % result[2]}_{str_time}.pdf"
        if not os.path.exists(f"outputs/pickle_list"):
            os.makedirs(f"outputs/pickle_list")
        f.savefig(file_name_plot_PDF, bbox_inches='tight')
        logging.info(f"The plot was saved into:  {file_name_plot_PDF}")
# detect_process_log(data_with_path, multiprocess=True, chunk=60)


# data_path="../data/benign/imdb_cleaned_all.csv"
# data_path="../data/benign/yelp_review_full_csv/mytest.csv"

name_file = "sus_glue"
# name_file = "imdb"
# df = pd.read_csv(data_path)['text'][:1000]

# avg = 0
# kk = 0
# for i in df:
#     avg += len(i)
#     kk +=1
# print(kk)
# print(avg/len(df))
# exit()


# print(df.shape)
# exit()

# df.columns = ['a', 'text']
# df.to_csv("../data/benign/yelp_review_full_csv/mytest.csv")
# df = pd.read_csv(data_path)["text"][:200]
# df = pd.read_csv(data_path)["sentence"]
# len_df = len(df)
len_df = 1000  # len(df)
time_str = time.strftime("%m-%d_%H-%M")
folder_name = f'{name_file}_{time_str}'
file_name_threshold = 'outputs/{}/{}_{}_{}_thresholds.txt'.format(folder_name, name_file, len_df,
                                                                  time_str)
file_name_k = 'outputs/{}/{}_{}_{}_k.txt'.format(folder_name, name_file, len_df, time_str)
file_name_plot_PDF = 'outputs/{}/{}_{}_{}_plot.pdf'.format(folder_name, name_file, len_df, time_str)

logging.info(data_path)

#####################################################
#####################--SETUP--#######################
#####################################################
TEST_THRESHOLD_WITH_TIME = False
SAVE_THRESHOLD_LIST = False
PLOT_THRESHOLD = False
SAVE_PLOT_THRESHOLD = False

TEST_DETECTION_MULTI = False
TEST_DETECTION_MULTI2 = False
TEST_DETECTION_SINGLE = False
#####################################################
#####################################################
#####################################################


k_ = 0
threshold = 0
if TEST_THRESHOLD_WITH_TIME:
    logging.info(f"data:::::  {name_file}")
    thd = Thresholds()
    start_time = time.time()

    k_, threshold = thd.calculate_thresholds(df, k=30, chunk=16, up_to_k=True,
                                             multiprocess="multiprocess", num_process=8, pair="pdist", percentile=0.1)
    elapsed_time = time.time() - start_time
    logging.info(f"time\t>>>>\t{elapsed_time} s")
    print("time\t>>>>>>>>\t\033[44m\033[31m", elapsed_time, " s \033[00m")
    #   k_, threshold = thd.calculate_thresholds(benign_queries[:100], k=30, chunk=10, up_to_k=True,
    #                                          multiprocess=True, num_process=4)
    print("\nK:\t\033[40m \033[31m", k_, "\033[00m\t<<<>>>\tthreshold:\t\033[40m \033[31m", threshold, "\033[00m")
    logging.info(f"K:{k_}    <<<>>>    threshold:{threshold}")
    results_thresholds = thd.list_of_thresholds
    results_of_k = thd.list_of_k
    # thd.__init__()
    if PLOT_THRESHOLD:
        from matplotlib import pyplot as plt

        f, ax = plt.subplots(1)
        ax.plot(results_of_k, results_thresholds)
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=0)
        plt.xlabel("k# of nearest neighbors")
        plt.ylabel("Threshold")
        # plt.plot(results_of_k, results_thresholds)
        plt.show()
        if SAVE_PLOT_THRESHOLD:
            if not os.path.exists(f"outputs/{folder_name}"):
                os.makedirs(f"outputs/{folder_name}")
            f.savefig(file_name_plot_PDF, bbox_inches='tight')
            logging.info(f"The plot was saved into:  {file_name_plot_PDF}")

    if SAVE_THRESHOLD_LIST:
        if not os.path.exists(f"outputs/{folder_name}"):
            os.makedirs(f"outputs/{folder_name}")
        with open(file_name_threshold, "wb") as fp:  # Pickling
            pickle.dump(results_thresholds, fp)
            logging.info(f"The list of thresholds was saved into:  {results_thresholds}")
        with open(file_name_k, "wb") as fp:  # Pickling
            pickle.dump(results_of_k, fp)
            logging.info(f"The list of k values was saved into:  {results_of_k}")

# df_sus = pd.concat([df.sample(frac=1).reset_index(drop=True)[:900],df[-100:]])
##################################################################################
if TEST_DETECTION_MULTI:
    detector = Detector(k=10, threshold=20)
    detector.multi_process(df, chunk=75)
    detector.print_result()
##################################################################################


if TEST_DETECTION_MULTI2:
    K_VALUE0 = 10
    THRESHOLD_VALUE0 = 0.21
    CHUNK_VALUE0 = 10000

    detector = Detector(k=K_VALUE0, threshold=THRESHOLD_VALUE0)
    logging.info(f"<<<<<<<<<<< <<<<<<<<<<<<<<<<  Detection TEST single - 2  >>>>>>>>>>>>>>>> >>>>>>>>>>>")
    logging.info(f"Detector(k={K_VALUE0}, theshold={THRESHOLD_VALUE0})")
    start_time = time.time()
    detector.multi_process(df_sus, chunk=CHUNK_VALUE0, sless=0.40)
    elapsed_time = time.time() - start_time
    logging.info(f"time\t>>>>\t{elapsed_time} s")
    logging.info(f"df_name: {name_file}  len(data)={len(df_sus)}  --- multi_process(df, )")
    epochs = detector.get_detections()
    num_attacks = len(epochs)
    logging.info(f"num of detections: {num_attacks}")
    logging.info(f"Queries per detection: {epochs}")
    logging.info(f"i-th query that caused detection: {detector.history}")
    logging.info(f"Num of queries: {detector.length_of_input}")
    logging.info(f"Avg. distance Detected:\t {detector.detected_dists}")
    print("all ", detector.history_attack, )

    detector.print_result()

if TEST_DETECTION_SINGLE:
    if True:
        K_VALUE1 = 10
        THRESHOLD_VALUE1 = 0.21
        detector = Detector(k=K_VALUE1, threshold=THRESHOLD_VALUE1)
        logging.info(f"<<<<<<<<<<< <<<<<<<<<<<<<<<  Detection TEST single - 1  >>>>>>>>>>>>>>> >>>>>>>>>>>")
        logging.info(f"Detector(k={K_VALUE1}, theshold={THRESHOLD_VALUE1})")
        start_time = time.time()
        detector.process(data_with_path[0])
        elapsed_time = time.time() - start_time
        logging.info(f"time\t>>>>\t{elapsed_time} s")

        logging.info(f"df_name: {data_with_path[1]}  len(data)={len(data_with_path[0])}  --- process(df,)")
        epochs = detector.get_detections()
        num_attacks = len(epochs)
        logging.info(f">>>>>> #num of detections: {num_attacks}")
        logging.info(f"Queries per detection: {epochs}")
        logging.info(f"i-th query that caused detection: {detector.history}")
        logging.info(f"Num of queries: {detector.length_of_input}")
        logging.info(f"Avg. distance Detected:\t {detector.detected_dists}")
        detector.print_result()
        print(f"time : : : {elapsed_time}")

        print("<<<<<<<<<<<<<<<<second>>>>>>>>>>>>>>>>")
        detector = Detector(k=K_VALUE1, threshold=THRESHOLD_VALUE1)
        start_time = time.time()
        detector.once_process(data_with_path[0])
        elapsed_time = time.time() - start_time
        logging.info(f"time\t>>>>\t{elapsed_time} s")

        logging.info(f"df_name: {data_with_path[1]}  len(data)={len(data_with_path[0])}  --- process(df,)")
        epochs = detector.get_detections()
        num_attacks = len(epochs)
        logging.info(f">>>>>> #num of detections: {num_attacks}")
        logging.info(f"Queries per detection: {epochs}")
        logging.info(f"i-th query that caused detection: {detector.history}")
        logging.info(f"Num of queries: {detector.length_of_input}")
        logging.info(f"Avg. distance Detected:\t {detector.detected_dists}")
        detector.print_result()
        print(f"time : : : {elapsed_time}")

# detector.multi_process(df_sus[:300], chunk=60, reset=True, sless=30)
# detector.print_result()
#
# detector.multi_process(df_sus, chunk=60, reset=True, sless=30)
# detector.print_result()


# print(detector.history_attack)


# ATTACK SETUP
# ##############
# cnn-yelp_deepwordbug_sequences_2020-11-20-15-34
# ATTACKER: deepwordbug
# GOAL FUNCTION: untargeted_classification.UntargetedClassification
# MODEL: WordCNNForClassification.
# [Succeeded / Failed / Total] 875 / 44 / 1000: 100% 1000/1000 [28:45<00:00,  1.73s/it]
# avg length: 1166.1961291182251
# number of sentence: 150095
# textattack: Loading pre-trained TextAttack CNN: cnn-yelp
# textattack: Goal function <class 'textattack.goal_functions.classification.untargeted_classification.UntargetedClassification'> compatible with model WordCNNForClassification.
# textattack: Logging sequences to CSV at path ./attacks/cnn-yelp_deepwordbug_sequences_2020-11-20-15-34.csv.
# textattack: Load time: 9.77456545829773s
# Reusing dataset yelp_polarity (/content/drive/My Drive/COLAB-GOOGLE/NLP-adversarial/QData/downloadedData/datasets/yelp_polarity/plain_text/1.0.0/2b33212d89209ed1ea0522001bccc5f5a5c920dd9c326f3c828e67a22c51a98c)
# textattack: Loading datasets dataset yelp_polarity, split test.
# [Succeeded / Failed / Total] 875 / 44 / 1000: 100% 1000/1000 [28:45<00:00,  1.73s/it]
# textattack: Attack time: 1731.6877031326294s
# -----
# DETECTION SETUP
# #################
# k=10, threshold=20
# Num of detections: 13406
# Average detection: 13.4


# print("Num detections:", len(detections))
# print("Queries per detection:", detections)
# print("i-th query that caused detection:", detector.history)
# Queries per detection: [31, 31, 58, 31, 31, 50, 47]
# i-th query that caused detection: [31, 62, 120, 151, 182, 232, 279]

# textattack: Downloading https://textattack.s3.amazonaws.com/models/classification/cnn/ag-news.
# 100% 298M/298M [00:08<00:00, 33.4MB/s]
# textattack: Unzipping file /content/drive/My Drive/COLAB-GOOGLE/NLP-adversarial/QData/downloadedData/tmpn2zm61f9.zip to /content/drive/My Drive/COLAB-GOOGLE/NLP-adversarial/QData/downloadedData/models/classification/cnn/ag-news.
# textattack: Successfully saved models/classification/cnn/ag-news to cache.
# textattack: Loading pre-trained TextAttack CNN: cnn-ag-news
# textattack: Goal function <class 'textattack.goal_functions.classification.untargeted_classification.UntargetedClassification'> compatible with model WordCNNForClassification.
# textattack: Logging sequences to CSV at path ./attacks/cnn-ag-news_deepwordbug_sequences_2020-11-30-18-29.csv.
# textattack: Load time: 39.28120255470276s
# Using custom data configuration default
# Reusing dataset ag_news (/content/drive/My Drive/COLAB-GOOGLE/NLP-adversarial/QData/downloadedData/datasets/ag_news/default/0.0.0/fb5c5e74a110037311ef5e904583ce9f8b9fbc1354290f97b4929f01b3f48b1a)
# textattack: Loading datasets dataset ag_news, split test.
# [Succeeded / Failed / Total] 898 / 10 / 1000: 100% 1000/1000 [06:20<00:00,  2.63it/s]
# textattack: Attack time: 381.4261591434479s


logging.info(f"#########  <<<<<<<<<<<<<<<<<<Finished>>>>>>>>>>>>>>>>>>  #########")


def create_data_series(number, path, new_name):
    df = pd.read_csv(path)
    df2 = df
    df.to_csv(f"1_{new_name}.csv")
    for i in range(2, number + 1):
        df2 = pd.concat([df, df2]).sample(frac=1).reset_index(drop=True)
        df2.to_csv(f"{i}_{new_name}.csv")
    print("finish")
