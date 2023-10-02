import csv

with open("predictions.csv") as file:
    csv_read = csv.reader(file, dialect="excel")
    for idx, lines in enumerate(csv_read):
        if idx == 10:
            break
        print("\n".join(lines))
