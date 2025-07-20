import csv
from datetime import datetime

feedback_file = "data/feedback_log.csv"

def log_feedback(query, result, relevance):
    with open(feedback_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([datetime.now(), query, result, relevance])