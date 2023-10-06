import json

import pandas as pd


def get_ranges(start_from: int, file_amt: int) -> list[int]:
    with open("indonesian_train.jsonl", encoding="utf-8") as file:
        rowcount = sum(1 for line in file)
    distribution = rowcount // file_amt
    dist_list = []
    for idx in range(file_amt + 1):
        dist_list.append(start_from + (distribution * idx))
    dist_list[-1] = rowcount
    return dist_list


def create_files(file_amt: int):
    dist_list = get_ranges(0, file_amt)
    print(dist_list)
    raw_data = []
    with open("indonesian_train.jsonl", encoding="utf-8") as file:
        for line in file:
            raw_data.append(json.loads(line))
        for idx, _ in enumerate(dist_list):
            if idx + 1 == len(dist_list):
                break
            data = {"text": [], "ctext": []}
            for dist in range(dist_list[idx], dist_list[idx + 1]):
                slices = raw_data[dist]
                data["text"].append(slices.get("summary"))
                data["ctext"].append(slices.get("text"))
            df = pd.DataFrame(data)
            df.to_csv(f"train_files/id_train_{idx}.csv", sep=",")
            data.clear()
        
create_files(100)