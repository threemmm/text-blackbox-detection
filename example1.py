from textdetection.detection import Detector, Thresholds
import pandas as pd
import time
import pickle
import logging
import os
from matplotlib import pyplot as plt

logging.basicConfig(filename='outputs/mylogs/mylogs.log', format='%(asctime)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logging.info(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< NEW START - Example1  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

data_path = "../data/suspicious/albert-base-v2-sst2_textbugger_sequences_2020-11-20-03-20.csv"  # text
data_with_path = []


def detect_process_log(data_with_path, k=10, threshold=0.2, multiprocess=False, chunk=20):
    detector = Detector(k=k, threshold=threshold)
    start_time = time.time()
    if multiprocess:
        detector.multi_process(data_with_path[0], chunk=chunk, sless=0)
    elif not multiprocess:
        detector.process(data_with_path[0])
        # detector.once_process(data_with_path[0]) #calculate pdist once (for large benign data)
    elapsed_time = time.time() - start_time
    logging.info(
        f"time\t>>>>\t{elapsed_time} s \tnum of detections: {len(detector.history)}\n\tdata: {data_with_path[1]}\t"
        f"len(data)={len(data_with_path[0])}\n\tDetector(k={k}, theshold={threshold}")
    detector.print_result()
    print(f"time elapsed : : : {elapsed_time}")
    return len(detector.history)


K = 15
THRESHOLD = 0.4
all_result = []
SAVE_PICKLE_PLOT = True

data_version = num_of_detection_list = []
file_name = "glue" #glue or yelp
# file_name = "yelp"
start_time = time.time()
# There are 15 glue dataset and 12 yelp
for i in range(4, 6):
    # data_path = f"../data/benign/yelp_review_full_csv/new_versions/{i}_{file_name}.csv"  # text
    data_path = f"../data/benign/glue_series/{i}_{file_name}.csv"  # column = "sentence"
    data_with_path = [pd.read_csv(data_path)['sentence'], data_path]
    num_of_detection = detect_process_log(data_with_path, k=K, threshold=THRESHOLD,
                                          multiprocess=False, chunk=60)
    print(f">>>>>>>>>>> {i} process finished.")
    data_version.append(i)
    num_of_detection_list.append(num_of_detection)
elapsed_time = time.time() - start_time
logging.info(f"<<<<<< time passed: {elapsed_time} >>>>>>")

result = [file_name, K, THRESHOLD, data_version, num_of_detection_list]
str_time = time.strftime("%m-%d_%H-%M")
if not os.path.exists(f"outputs/pickle_list"):
    os.makedirs(f"outputs/pickle_list")
file_name_pickle = f"outputs/pickle_list/pickle_{str_time}.txt"
with open(file_name_pickle, "wb") as fp:  # Pickling
    pickle.dump(result, fp)
logging.info(f"The list of results was saved into: {file_name_pickle}")

if SAVE_PICKLE_PLOT:
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

logging.info(f"#########  <<<<<<<<<<<<<<<<<<Finished>>>>>>>>>>>>>>>>>>  #########")

#
# def create_data_series(number, path, new_name):
#     df = pd.read_csv(path)
#     df2 = df
#     df.to_csv(f"1_{new_name}.csv")
#     for i in range(2, number + 1):
#         df2 = pd.concat([df, df2]).sample(frac=1).reset_index(drop=True)
#         df2.to_csv(f"{i}_{new_name}.csv")
#     print("finish")
