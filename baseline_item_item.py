from pathlib import Path
import csv
import math
import torch

DATA_DIR = Path(r"D:\DOANHOCSAU")
TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "test.csv"

NUM_USERS = 943
NUM_MOVIES = 1682
TOP_K = 10
SHRINKAGE = 10.0

# Tải dữ liệu train
train_rows = []
with TRAIN_FILE.open("r", newline="", encoding="utf-8") as file:
    for row in csv.DictReader(file):
        train_rows.append((
            int(row["user_id"]),
            int(row["movie_id"]),
            int(row["rating"]),
            int(row["label"]),
        ))

# Tạo ma trận phim × người dùng: 1 nếu user thích phim trong train
item_user = torch.zeros((NUM_MOVIES, NUM_USERS), dtype=torch.float32)
seen_by_user = [set() for _ in range(NUM_USERS)]
liked_by_user = [set() for _ in range(NUM_USERS)]
movie_popularity = [0] * NUM_MOVIES

for user_id, movie_id, rating, label in train_rows:
    seen_by_user[user_id].add(movie_id)
    if label == 1:
        item_user[movie_id, user_id] = 1.0
        liked_by_user[user_id].add(movie_id)
        movie_popularity[movie_id] += 1

# Cosine similarity giữa các phim dựa trên người dùng cùng thích chúng
co_likes = item_user @ item_user.T
norms = torch.sqrt(torch.diag(co_likes)).clamp(min=1e-8)
similarity = co_likes / (norms[:, None] * norms[None, :])

# Giảm ảnh hưởng của các cặp phim chỉ có ít người dùng chung
similarity *= co_likes / (co_likes + SHRINKAGE)
similarity.fill_diagonal_(0.0)

# Đọc các phim được thích trong validation
relevant_by_user = {}
with VALIDATION_FILE.open("r", newline="", encoding="utf-8") as file:
    for row in csv.DictReader(file):
        user_id = int(row["user_id"])
        movie_id = int(row["movie_id"])
        label = int(row["label"])
        if label == 1:
            relevant_by_user.setdefault(user_id, set()).add(movie_id)

def ranking_metrics(recommendations, relevant):
    hits = sum(movie_id in relevant for movie_id in recommendations)
    precision = hits / TOP_K
    recall = hits / len(relevant)

    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, movie_id in enumerate(recommendations)
        if movie_id in relevant
    )
    ideal_hits = min(len(relevant), TOP_K)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    ndcg = dcg / idcg if idcg else 0.0

    return precision, recall, ndcg

item_metrics = []
pop_metrics = []

for user_id, relevant in relevant_by_user.items():
    seen = seen_by_user[user_id]
    liked = liked_by_user[user_id]

    candidates = [
        movie_id for movie_id in range(NUM_MOVIES)
        if movie_id not in seen
    ]

    # Điểm Item-Item: tổng độ tương tự với các phim user đã thích
    if liked:
        profile = torch.zeros(NUM_MOVIES, dtype=torch.float32)
        profile[list(liked)] = 1.0
        scores = profile @ similarity
        item_top = sorted(
            candidates,
            key=lambda movie_id: float(scores[movie_id]),
            reverse=True,
        )[:TOP_K]
    else:
        item_top = sorted(
            candidates,
            key=lambda movie_id: movie_popularity[movie_id],
            reverse=True,
        )[:TOP_K]

    pop_top = sorted(
        candidates,
        key=lambda movie_id: movie_popularity[movie_id],
        reverse=True,
    )[:TOP_K]

    item_metrics.append(ranking_metrics(item_top, relevant))
    pop_metrics.append(ranking_metrics(pop_top, relevant))

def print_average(name, metrics):
    count = len(metrics)
    precision = sum(x[0] for x in metrics) / count
    recall = sum(x[1] for x in metrics) / count
    ndcg = sum(x[2] for x in metrics) / count

    print(f"\n{name} — validation trên {count} user:")
    print(f"Precision@{TOP_K}: {precision:.4f}")
    print(f"Recall@{TOP_K}:    {recall:.4f}")
    print(f"NDCG@{TOP_K}:      {ndcg:.4f}")

print_average("Item-Item Collaborative Filtering", item_metrics)
print_average("Baseline phim phổ biến", pop_metrics)