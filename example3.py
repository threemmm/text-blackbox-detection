from textdetection.detection import Detector
import pandas as pd
import time
import pickle
import logging

import os
from matplotlib import pyplot as plt

logging.basicConfig(filename='outputs/mylogs/mylogs.log', format='%(asctime)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logging.info(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< NEW START  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')


def detect_process_log(data_with_path, k, threshold, multiprocess=False, chunk=20):
    detector = Detector(k=k, threshold=threshold)
    # logging.info(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<  detect_process_log  >>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # logging.info(f"Detector(k={k}, theshold={threshold})")
    start_time = time.time()
    if multiprocess:
        detector.multi_process(data_with_path[0], chunk=chunk, sless=0)
    elif not multiprocess:
        detector.process(data_with_path[0])
    elapsed_time = time.time() - start_time
    logging.info(
        f"time\t>>>>\t{elapsed_time} s \n\tdata: {data_with_path[1]}\t"
        f"len(data)={len(data_with_path[0])}\n\tDetector(k={k}, theshold={threshold})"
        f"\tNum of detections >>> {len(detector.history)} <<<\n\t positions(i-th): {detector.history}\n--------------")
    detector.print_result()
    print(f"time elapsed : : : {elapsed_time}")
    return detector


df_list_yelp = []
K = 5
THRESHOLD = 0.1
all_result = []
SAVE_ALL_RESULTS = False
SAVE_PICKLE_PLOT = False

list1 = ["../data/newdatasets_glue/glue_4replica.csv", "../data/newdatasets_glue/glue_5replica.csv",
         "../data/newdatasets_glue/glue_6replica.csv"]
list2 = ["../data/newdatasets_glue/first10attack1960_cnn-yelp_textbugger.csv",
         "../data/newdatasets_glue/first10attack1650_cnn-yelp_deepwordbug.csv"]
k_values = []
num_of_detection_list = []
thresholdsss = []
file_name = f"glue_replica"
start_time = time.time()
# data_path = f"test_for_attack_30.csv"
# data_path = "../data/newdatasets_glue/glue_4replica.csv" #sentence
# data_path = "../data/newdatasets_glue/glue_6replica.csv" #sentence
# data_path = "../data/newdatasets_glue/glue_7replica.csv" #sentence
data_path = "../data/newdatasets_glue/glue_8replica.csv"  # sentence
data_path_oneattack = "../data/newdatasets_glue/one_attack_204_cnn-yelp_textbugger.csv"
data_path_oneattack = "../data/newdatasets_glue/one_attack_153_cnn-yelp_deepwordbug.csv"
# data_path_oneattack = "../data/newdatasets_glue/first10attack1960_cnn-yelp_textbugger.csv"
# data_path_oneattack = "../data/newdatasets_glue/first10attack1650_cnn-yelp_deepwordbug.csv"
df_sum = pd.concat([pd.read_csv(data_path_oneattack)['text'], pd.read_csv(data_path)['text']])
data_with_path = [df_sum, data_path_oneattack + "+" + data_path]
k = [8, 10, 15, 15, 20, 20, 25]
thr = [0.12, 0.21, 0.21, 0.35, 0.25, 0.40, 0.30]
k = [10, 20]
thr = [0.10, 0.20]
# for zz in [25]:
# for i in [0.25]:
for i, zz in zip(thr, k):
    print("k:", zz, "thr:", i)
    K = zz
    THRESHOLD = i
    detector_inside = detect_process_log(data_with_path, k=K, threshold=THRESHOLD,
                                         multiprocess=True, chunk=350)
    # for ii in detector_inside.history:
    #     print(f"\nindex {ii}\n",data_with_path[0].iloc[ii])
    print(f">>>>>>>>>>> {i} process finished.")
    k_values.append(K)
    num_of_detection_list.append(len(detector_inside.history))
    thresholdsss.append(THRESHOLD)
    # THRESHOLD += 0.02
    # K += 2
elapsed_time = time.time() - start_time
logging.info(f"<<<<<<<<<<<<<<< time passed: {elapsed_time} >>>>>>>>>>>>>>>")

# K += 1
# THRESHOLD += 0.01
result = [file_name, K, THRESHOLD, k_values, thresholdsss, num_of_detection_list]
if SAVE_ALL_RESULTS:
    if not os.path.exists(f"outputs/pickle_list"):
        os.makedirs(f"outputs/pickle_list")
    str_time = time.strftime("%m-%d_%H-%M")
    file_name_pickle = f"outputs/pickle_list/pickle_{str_time}.txt"
    with open(file_name_pickle, "wb") as fp:  # Pickling
        pickle.dump(result, fp)
    logging.info(f"The list of results was saved into: {file_name_pickle}")

if SAVE_PICKLE_PLOT:
    # str_time = time.strftime("%m-%d_%H-%M-%s")
    f, ax = plt.subplots(1)
    ax.plot(result[3], result[5])
    # ax.set_ylim(ymin=0)
    # ax.set_xlim(xmin=0)
    plt.xlabel(f"K")
    plt.ylabel("#Num of detections")
    plt.title(f"threshold={THRESHOLD}")
    # plt.plot(results_of_k, results_thresholds)
    plt.show()

    file_name_plot_PDF = f"outputs/pickle_list/plot_k_{result[1]}_threshold_{'%.2f' % result[2]}_{str_time}.pdf"
    if not os.path.exists(f"outputs/pickle_list"):
        os.makedirs(f"outputs/pickle_list")
    f.savefig(file_name_plot_PDF, bbox_inches='tight')
    logging.info(f"The plot was saved into:  {file_name_plot_PDF}")


def counting(list1, k):
    count = len([i for i in list1 if i > k])
    print("error :", count)
    print("correct:  ", len(list1) - count)
