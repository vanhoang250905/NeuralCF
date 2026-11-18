from pathlib import Path
import csv
import random

source = Path(r"D:\DOANHOCSAU\interactions.csv")
data_dir = Path(r"D:\DOANHOCSAU\ml-100k\ml-100k")
train_path = Path(r"D:\DOANHOCSAU\train.csv")
test_path = Path(r"D:\DOANHOCSAU\test.csv")
movie_map_path = Path(r"D:\DOANHOCSAU\movie_map.csv")

random.seed(42)

with source.open("r", newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

# Đọc tên phim từ MovieLens u.item
titles = {}
with (data_dir / "u.item").open("r", encoding="latin-1") as file:
    for line in file:
        fields = line.rstrip("\n").split("|")
        if len(fields) >= 2:
            titles[fields[0]] = fields[1]

# Mã hóa ID thành số liên tục từ 0
user_ids = sorted({row["user_id"] for row in rows}, key=int)
movie_ids = sorted({row["movie_id"] for row in rows}, key=int)
user_map = {old_id: new_id for new_id, old_id in enumerate(user_ids)}
movie_map = {old_id: new_id for new_id, old_id in enumerate(movie_ids)}

# Lưu bảng tra ID phim đã mã hóa -> ID gốc -> tên phim
with movie_map_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["encoded_movie_id", "original_movie_id", "title"])
    for original_id, encoded_id in movie_map.items():
        writer.writerow([encoded_id, original_id, titles.get(original_id, "Unknown")])

for row in rows:
    row["user_id"] = user_map[row["user_id"]]
    row["movie_id"] = movie_map[row["movie_id"]]
    row["rating"] = int(row["rating"])
    row["label"] = int(row["label"])

random.shuffle(rows)

# Giữ 20% làm test
test_index = int(len(rows) * 0.8)
train_validation_rows = rows[:test_index]
test_rows = rows[test_index:]

# Lấy 20% tổng dữ liệu làm validation
validation_index = int(len(train_validation_rows) * 0.25)
validation_rows = train_validation_rows[:validation_index]
train_rows = train_validation_rows[validation_index:]

columns = ["user_id", "movie_id", "rating", "label"]

for filename, data in [
    ("train.csv", train_rows),
    ("validation.csv", validation_rows),
    ("test.csv", test_rows),
]:
    path = Path(r"D:\DOANHOCSAU") / filename
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(data)
    print(f"{filename}: {len(data)} dòng → {path}")

print(f"Số user: {len(user_map)}, số phim: {len(movie_map)}")
print(f"Bảng tên phim: {movie_map_path}")