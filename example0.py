from textdetection.detection import Detector
import pandas as pd

data_path = "../data/suspicious/albert-base-v2-sst2_textbugger_sequences_2020-11-20-03-20.csv"  # text

dataset = pd.read_csv(data_path)
text_column = dataset['text']

detector = Detector(k=10, threshold=0.21)
detector.process(text_column)
# detector.multi_process(text_column, chunk=100)
detector.print_result()

# access to the number of detections and their positions
print(f"\n\nNum of detections: {len(detector.history)}")
print(f"i-th query that caused detections: {detector.history}")
