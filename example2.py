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


def detect_process_log(data_with_path, k=10, threshold=0.4, multiprocess=False, chunk=20):
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
        f"time\t>>>>\t{elapsed_time} s \tnum of detections: {len(detector.history)}\n\tdata: {data_with_path[1]}\t"
        f"len(data)={len(data_with_path[0])}\n\tDetector(k={k}, theshold={threshold} \n\t positions: {detector.history}")
    detector.print_result()
    print(f"time elapsed : : : {elapsed_time}")
    return detector


df_list_yelp = []
K = 5
THRESHOLD = 0.1
all_result = []
SAVE_ALL_RESULTS = False
SAVE_PICKLE_PLOT = False
if True:
    k_values = []
    num_of_detection_list = []
    thresholdsss = []
    file_name = f"glue"
    start_time = time.time()
    data_path = f"test_for_attack_30.csv"
    data_path = "../data/benign/glue.csv"  # sentence
    data_path = "../data/benign/yelp_review_full_csv/mytest.csv"
    # for i in np.linspace(0.1, 0.4, num=20):
    data_with_path = [pd.read_csv(data_path)['text'][:1000], data_path]
    for z in [10]:  # , 50, 60]: #[10, 11, 12, 13, 14, 15]:
        for i in [0.41, 0.42, 0.44, 0.46]:  # [0.1, 0.2, 0.3, 0.4, 0.5]:#, 0.17, 0.22, 0.27, 0.3]:
            # K = random.randint(6, 30)
            K = z
            THRESHOLD = i
            # file_name = f"yelp"
            # data_path = f"../data/benign/yelp_review_full_csv/new_versions/{i}_" + file_name + ".csv"#text
            # data_path = f"../data/benign/glue_series/{i}_" + file_name + ".csv"#text
            # df_list_yelp.append([pd.read_csv(data_path)['text'], data_path])
            # data_with_path = [pd.read_csv(data_path)['sentence'], data_path]
            detector_inside = detect_process_log(data_with_path, k=K, threshold=THRESHOLD,
                                                 multiprocess=True, chunk=80)
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

#
# def Insert_row(row_number, df, row_value):
#     # Starting value of upper half
#     start_upper = 0
#
#     # End value of upper half
#     end_upper = row_number
#
#     # Start value of lower half
#     start_lower = row_number
#
#     # End value of lower half
#     end_lower = df.shape[0]
#
#     # Create a list of upper_half index
#     upper_half = [*range(start_upper, end_upper, 1)]
#
#     # Create a list of lower_half index
#     lower_half = [*range(start_lower, end_lower, 1)]
#
#     # Increment the value of lower half by 1
#     lower_half = [x.__add__(1) for x in lower_half]
#
#     # Combine the two lists
#     index_ = upper_half + lower_half
#
#     # Update the index of the dataframe
#     df.index = index_
#
#     # Insert a row at the end
#     df.loc[row_number] = row_value
#
#     # Sort the index labels
#     df = df.sort_index()
#
#     # return the dataframe
#     return df
