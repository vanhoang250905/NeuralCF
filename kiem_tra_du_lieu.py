from pathlib import Path

data_path = Path(r"D:\DOANHOCSAU\ml-100k\ml-100k\u.data")

if not data_path.exists():
    raise FileNotFoundError(f"Không tìm thấy tệp: {data_path}")

users = set()
movies = set()
ratings = []
count = 0

with data_path.open("r", encoding="latin-1") as file:
    for line in file:
        fields = line.split()

        if len(fields) != 4:
            continue

        user_id, movie_id, rating, timestamp = fields

        users.add(user_id)
        movies.add(movie_id)
        ratings.append(int(rating))
        count += 1

print("Số lượt đánh giá:", count)
print("Số người dùng:", len(users))
print("Số phim:", len(movies))
print("Điểm thấp nhất:", min(ratings))
print("Điểm cao nhất:", max(ratings))

print("\n5 dòng đầu:")
with data_path.open("r", encoding="latin-1") as file:
    for _, line in zip(range(5), file):
        print(line.strip())