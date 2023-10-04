import json

import pandas as pd

def main():
    filenames = ("test", "train", "val")
    for name in filenames:
        training_data = {"text": [], "ctext": []}
        with open(f"indonesian_{name}.jsonl", encoding='utf-8') as file:
            for idx, line in enumerate(file):
                text = json.loads(line)
                training_data["ctext"].append(text.get("title")+ " " + text.get("text"))
                training_data["text"].append(text.get("summary"))
        
        df = pd.DataFrame(training_data)
        df.to_csv(f"id_{name}.csv", encoding="utf-8")

if __name__ == '__main__':
    main()