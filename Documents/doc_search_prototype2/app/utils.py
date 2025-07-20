import json
import os

def load_feedback(path="feedback_log.json"):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)

def save_feedback(feedback, path="feedback_log.json"):
    with open(path, "w") as f:
        json.dump(feedback, f, indent=4)

def filter_no_feedback(results, query, feedback):
    """Exclude previously marked 'No' items from being shown again for this query"""
    no_ids = feedback.get(query, {}).get("no", [])
    return [r for r in results if r['id'] not in no_ids]
