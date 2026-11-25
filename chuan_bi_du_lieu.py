from pathlib import Path
import csv

source = Path(r"D:\DOANHOCSAU\ml-100k\ml-100k\u.data")
output = Path(r"D:\DOANHOCSAU\interactions.csv")

with source.open("r", encoding="latin-1") as infile, \
     output.open("w", newline="", encoding="utf-8") as outfile:

    writer = csv.writer(outfile)
    writer.writerow(["user_id", "movie_id", "rating", "label"])

    for line in infile:
        user_id, movie_id, rating, timestamp = line.split()
        rating = int(rating)
        label = 1 if rating >= 4 else 0
        writer.writerow([user_id, movie_id, rating, label])

print(f"Đã tạo: {output}")
print("Quy ước nhãn: rating >= 4 là thích (1), rating < 4 là không thích (0)")