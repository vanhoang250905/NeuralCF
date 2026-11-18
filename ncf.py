from pathlib import Path
import csv
import random
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Cố định kết quả để các lần chạy dễ so sánh
random.seed(42)
torch.manual_seed(42)

DATA_DIR = Path(r"D:\DOANHOCSAU")
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
MOVIE_MAP_FILE = DATA_DIR / "movie_map.csv"

NUM_USERS = 943
NUM_MOVIES = 1682
EMBEDDING_SIZE = 32
EPOCHS = 10
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
USER_ID = 10       # Đổi thành user từ 0 đến 942
TOP_K = 10

device = torch.device("cpu")


def load_interactions(path):
    users, movies, labels = [], [], []

    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            users.append(int(row["user_id"]))
            movies.append(int(row["movie_id"]))
            labels.append(float(row["label"]))

    return (
        torch.tensor(users, dtype=torch.long),
        torch.tensor(movies, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )


class NCF(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size):
        super().__init__()

        # Embedding riêng cho nhánh GMF
        self.user_gmf = nn.Embedding(num_users, embedding_size)
        self.movie_gmf = nn.Embedding(num_movies, embedding_size)

        # Embedding riêng cho nhánh MLP
        self.user_mlp = nn.Embedding(num_users, embedding_size)
        self.movie_mlp = nn.Embedding(num_movies, embedding_size)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Ghép đầu ra GMF và MLP, cho ra một điểm dự đoán
        self.output = nn.Linear(embedding_size + 32, 1)

    def forward(self, user_ids, movie_ids):
        gmf_vector = (
            self.user_gmf(user_ids) * self.movie_gmf(movie_ids)
        )

        mlp_input = torch.cat(
            [self.user_mlp(user_ids), self.movie_mlp(movie_ids)],
            dim=1,
        )
        mlp_vector = self.mlp(mlp_input)

        combined = torch.cat([gmf_vector, mlp_vector], dim=1)
        return self.output(combined).squeeze(1)


train_users, train_movies, train_labels = load_interactions(TRAIN_FILE)
test_users, test_movies, test_labels = load_interactions(TEST_FILE)

train_dataset = TensorDataset(train_users, train_movies, train_labels)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

model = NCF(NUM_USERS, NUM_MOVIES, EMBEDDING_SIZE).to(device)
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Huấn luyện và đánh giá accuracy trên test
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    total_rows = 0

    for users, movies, labels in train_loader:
        users = users.to(device)
        movies = movies.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(users, movies)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        total_rows += len(labels)

    model.eval()
    with torch.no_grad():
        test_logits = model(test_users, test_movies)
        predictions = (torch.sigmoid(test_logits) >= 0.5).float()
        accuracy = (predictions == test_labels).float().mean().item()

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train loss: {total_loss / total_rows:.4f} | "
        f"Test accuracy : {accuracy:.4f}"
    )
# Lưu mô hình đã huấn luyện để app tải lại, không huấn luyện mỗi lần mở web
model_path = DATA_DIR / "ncf_model.pth"

torch.save(
    {
        "state_dict": model.state_dict(),
        "num_users": NUM_USERS,
        "num_movies": NUM_MOVIES,
        "embedding_size": EMBEDDING_SIZE,
    },
    model_path,
)

print(f"Đã lưu NCF: {model_path}")
# Đọc tên phim theo ID đã mã hóa
movie_titles = {}
with MOVIE_MAP_FILE.open("r", newline="", encoding="utf-8") as file:
    for row in csv.DictReader(file):
        movie_titles[int(row["encoded_movie_id"])] = row["title"]

# Bỏ qua phim user đã có trong tập train
watched = {
    int(movie_id)
    for user_id, movie_id in zip(train_users.tolist(), train_movies.tolist())
    if user_id == USER_ID
}

# Tính điểm cho các phim chưa xem
candidate_movies = [
    movie_id for movie_id in range(NUM_MOVIES)
    if movie_id not in watched
]

model.eval()
with torch.no_grad():
    user_tensor = torch.full(
        (len(candidate_movies),),
        USER_ID,
        dtype=torch.long,
    )
    movie_tensor = torch.tensor(candidate_movies, dtype=torch.long)
    scores = model(user_tensor, movie_tensor)

ranked = sorted(
    zip(scores.tolist(), candidate_movies),
    reverse=True,
)[:TOP_K]

print(f"\nTop {TOP_K} phim gợi ý cho user {USER_ID}:")
for rank, (score, movie_id) in enumerate(ranked, start=1):
    title = movie_titles.get(movie_id, "Không tìm thấy tên phim")
    print(f"{rank}. {title} | điểm mô hình: {score:.4f}")

# Đánh giá xếp hạng trung bình trên tất cả user có phim thích trong test
train_movies_by_user = {}
for user_id, movie_id in zip(train_users.tolist(), train_movies.tolist()):
    train_movies_by_user.setdefault(user_id, set()).add(movie_id)

relevant_by_user = {}
for user_id, movie_id, label in zip(
    test_users.tolist(),
    test_movies.tolist(),
    test_labels.tolist(),
):
    if label == 1:
        relevant_by_user.setdefault(user_id, set()).add(movie_id)

precision_scores = []
recall_scores = []
ndcg_scores = []

model.eval()

for eval_user_id, relevant_movies in relevant_by_user.items():
    seen_movies = train_movies_by_user.get(eval_user_id, set())
    candidates = [
        movie_id for movie_id in range(NUM_MOVIES)
        if movie_id not in seen_movies
    ]

    if not candidates:
        continue

    with torch.no_grad():
        users_tensor = torch.full(
            (len(candidates),),
            eval_user_id,
            dtype=torch.long,
        )
        movies_tensor = torch.tensor(candidates, dtype=torch.long)
        scores = model(users_tensor, movies_tensor).tolist()

    top_movies = [
        movie_id
        for _, movie_id in sorted(
            zip(scores, candidates),
            reverse=True,
        )[:TOP_K]
    ]

    hits = sum(movie_id in relevant_movies for movie_id in top_movies)
    precision_scores.append(hits / TOP_K)
    recall_scores.append(hits / len(relevant_movies))

    dcg = sum(
        1.0 / torch.log2(torch.tensor(rank + 2.0)).item()
        for rank, movie_id in enumerate(top_movies)
        if movie_id in relevant_movies
    )
    ideal_hits = min(len(relevant_movies), TOP_K)
    idcg = sum(
        1.0 / torch.log2(torch.tensor(rank + 2.0)).item()
        for rank in range(ideal_hits)
    )
    ndcg_scores.append(dcg / idcg if idcg else 0.0)

print(f"\nĐánh giá trung bình trên {len(precision_scores)} user:")
print(f"Precision@{TOP_K}: {sum(precision_scores) / len(precision_scores):.4f}")
print(f"Recall@{TOP_K}:    {sum(recall_scores) / len(recall_scores):.4f}")
print(f"NDCG@{TOP_K}:      {sum(ndcg_scores) / len(ndcg_scores):.4f}")

# Baseline: xếp phim theo tỷ lệ lượt thích trong tập train
movie_positive = [0] * NUM_MOVIES
movie_total = [0] * NUM_MOVIES

for movie_id, label in zip(train_movies.tolist(), train_labels.tolist()):
    movie_total[movie_id] += 1
    if label == 1:
        movie_positive[movie_id] += 1

# Dùng tỷ lệ thích để tránh ưu tiên phim chỉ có một vài lượt đánh giá
movie_like_rate = [
    movie_positive[i] / movie_total[i] if movie_total[i] else 0.0
    for i in range(NUM_MOVIES)
]

baseline_precision = []
baseline_recall = []
baseline_ndcg = []

for eval_user_id, relevant_movies in relevant_by_user.items():
    seen_movies = train_movies_by_user.get(eval_user_id, set())

    candidates = [
        movie_id for movie_id in range(NUM_MOVIES)
        if movie_id not in seen_movies
    ]

    top_movies = sorted(
        candidates,
        key=lambda movie_id: movie_like_rate[movie_id],
        reverse=True,
    )[:TOP_K]

    hits = sum(movie_id in relevant_movies for movie_id in top_movies)
    baseline_precision.append(hits / TOP_K)
    baseline_recall.append(hits / len(relevant_movies))

    dcg = sum(
        1.0 / torch.log2(torch.tensor(rank + 2.0)).item()
        for rank, movie_id in enumerate(top_movies)
        if movie_id in relevant_movies
    )
    ideal_hits = min(len(relevant_movies), TOP_K)
    idcg = sum(
        1.0 / torch.log2(torch.tensor(rank + 2.0)).item()
        for rank in range(ideal_hits)
    )
    baseline_ndcg.append(dcg / idcg if idcg else 0.0)

print("\nBaseline phim phổ biến (tỷ lệ thích trong train):")
print(f"Precision@{TOP_K}: {sum(baseline_precision) / len(baseline_precision):.4f}")
print(f"Recall@{TOP_K}:    {sum(baseline_recall) / len(baseline_recall):.4f}")
print(f"NDCG@{TOP_K}:      {sum(baseline_ndcg) / len(baseline_ndcg):.4f}")

# Baseline 2: xếp phim theo tổng số lượt thích trong train
count_precision = []
count_recall = []
count_ndcg = []

for eval_user_id, relevant_movies in relevant_by_user.items():
    seen_movies = train_movies_by_user.get(eval_user_id, set())

    candidates = [
        movie_id for movie_id in range(NUM_MOVIES)
        if movie_id not in seen_movies
    ]

    top_movies = sorted(
        candidates,
        key=lambda movie_id: movie_positive[movie_id],
        reverse=True,
    )[:TOP_K]

    hits = sum(movie_id in relevant_movies for movie_id in top_movies)
    count_precision.append(hits / TOP_K)
    count_recall.append(hits / len(relevant_movies))

    dcg = sum(
        1.0 / torch.log2(torch.tensor(rank + 2.0)).item()
        for rank, movie_id in enumerate(top_movies)
        if movie_id in relevant_movies
    )
    ideal_hits = min(len(relevant_movies), TOP_K)
    idcg = sum(
        1.0 / torch.log2(torch.tensor(rank + 2.0)).item()
        for rank in range(ideal_hits)
    )
    count_ndcg.append(dcg / idcg if idcg else 0.0)

print("\nBaseline phim phổ biến (số lượt thích trong train):")
print(f"Precision@{TOP_K}: {sum(count_precision) / len(count_precision):.4f}")
print(f"Recall@{TOP_K}:    {sum(count_recall) / len(count_recall):.4f}")
print(f"NDCG@{TOP_K}:      {sum(count_ndcg) / len(count_ndcg):.4f}")


# Thử kết hợp điểm NCF với độ phổ biến của phim trên validation
alphas = [32.0]
max_popularity = max(movie_positive) or 1

hybrid_results = {
    alpha: {"precision": [], "recall": [], "ndcg": []}
    for alpha in alphas
}

model.eval()

for eval_user_id, relevant_movies in relevant_by_user.items():
    seen_movies = train_movies_by_user.get(eval_user_id, set())
    candidates = [
        movie_id for movie_id in range(NUM_MOVIES)
        if movie_id not in seen_movies
    ]

    with torch.no_grad():
        users_tensor = torch.full(
            (len(candidates),),
            eval_user_id,
            dtype=torch.long,
        )
        movies_tensor = torch.tensor(candidates, dtype=torch.long)
        ncf_scores = model(users_tensor, movies_tensor).tolist()

    # Chuẩn hóa log số lượt thích về khoảng 0–1
    popularity_scores = [
        math.log1p(movie_positive[movie_id])
        / math.log1p(max_popularity)
        for movie_id in candidates
    ]

    for alpha in alphas:
        combined = [
            (ncf_score + alpha * popularity, movie_id)
            for ncf_score, popularity, movie_id
            in zip(ncf_scores, popularity_scores, candidates)
        ]
        top_movies = [
            movie_id
            for _, movie_id in sorted(combined, reverse=True)[:TOP_K]
        ]

        hits = sum(movie_id in relevant_movies for movie_id in top_movies)
        hybrid_results[alpha]["precision"].append(hits / TOP_K)
        hybrid_results[alpha]["recall"].append(
            hits / len(relevant_movies)
        )

        dcg = sum(
            1.0 / math.log2(rank + 2)
            for rank, movie_id in enumerate(top_movies)
            if movie_id in relevant_movies
        )
        ideal_hits = min(len(relevant_movies), TOP_K)
        idcg = sum(
            1.0 / math.log2(rank + 2)
            for rank in range(ideal_hits)
        )
        hybrid_results[alpha]["ndcg"].append(
            dcg / idcg if idcg else 0.0
        )

print("\nNCF kết hợp độ phổ biến — kết quả test:")
for alpha in alphas:
    result = hybrid_results[alpha]
    p = sum(result["precision"]) / len(result["precision"])
    r = sum(result["recall"]) / len(result["recall"])
    n = sum(result["ndcg"]) / len(result["ndcg"])
    print(
        f"alpha={alpha:.2f} | "
        f"Precision@{TOP_K}: {p:.4f} | "
        f"Recall@{TOP_K}: {r:.4f} | "
        f"NDCG@{TOP_K}: {n:.4f}"
    )

best_alpha = max(
    alphas,
    key=lambda alpha: sum(hybrid_results[alpha]["ndcg"])
    / len(hybrid_results[alpha]["ndcg"]),
)
print(f"Alpha đã chọn trên validation: {alphas[0]:.2f}")
