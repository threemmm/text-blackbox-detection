from textdetection.detection import Thresholds
import pandas as pd
import time
import pickle
import logging
import os
from matplotlib import pyplot as plt


def main():
    data_path = "../data/benign/imdb_cleaned_all.csv"
    # data_path="../data/benign/yelp_review_full_csv/mytest.csv"
    len_df = 1000  # 10000 for imdb and 15000 for yelp
    name_file = "imdb"
    df = pd.read_csv(data_path)['text'][:len_df]

    # df = pd.read_csv(data_path)["sentence"]
    time_str = time.strftime("%m-%d_%H-%M")
    folder_name = f'{name_file}_{time_str}'
    file_name_threshold = 'outputs/{}/{}_{}_{}_thresholds.txt'.format(folder_name, name_file, len_df,
                                                                      time_str)
    file_name_k = 'outputs/{}/{}_{}_{}_k.txt'.format(folder_name, name_file, len_df, time_str)
    file_name_plot_PDF = 'outputs/{}/{}_{}_{}_plot.pdf'.format(folder_name, name_file, len_df, time_str)

    logging.info(data_path)

    k_ = 0
    threshold = 0
    logging.info(f"data:::::  {name_file}")
    thd = Thresholds()
    start_time = time.time()

    k_, threshold = thd.calculate_thresholds(df, k=30, chunk=16, up_to_k=True,
                                             multiprocess="pooling", num_process=8, pair="pdist", percentile=0.1)
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
    f, ax = plt.subplots(1)
    ax.plot(results_of_k, results_thresholds)
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    plt.xlabel("k# of nearest neighbors")
    plt.ylabel("Threshold")
    plt.plot(results_of_k, results_thresholds)
    plt.show()

    if not os.path.exists(f"outputs/{folder_name}"):
        os.makedirs(f"outputs/{folder_name}")
    f.savefig(file_name_plot_PDF, bbox_inches='tight')
    logging.info(f"The plot was saved into:  {file_name_plot_PDF}")

    if not os.path.exists(f"outputs/{folder_name}"):
        os.makedirs(f"outputs/{folder_name}")
    with open(file_name_threshold, "wb") as fp:  # Pickling
        pickle.dump(results_thresholds, fp)
        logging.info(f"The list of thresholds was saved into:  {results_thresholds}")
    with open(file_name_k, "wb") as fp:  # Pickling
        pickle.dump(results_of_k, fp)
        logging.info(f"The list of k values was saved into:  {results_of_k}")


if __name__ == '__main__':
    main()
