from pathlib import Path

import csv
import math
import random
import csv

random.seed(42)

TRAIN_FILE = Path(r"D:\DOANHOCSAU\train.csv")
TEST_FILE = Path(r"D:\DOANHOCSAU\test.csv")
NUM_USERS = 943
NUM_MOVIES = 1682
EMBEDDING_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.03


def load_csv(path):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows.append((
                int(row["user_id"]),
                int(row["movie_id"]),
                int(row["label"])
            ))
    return rows


def sigmoid(value):
    # Tránh tràn số khi tính exp
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-value))


train_data = load_csv(TRAIN_FILE)
test_data = load_csv(TEST_FILE)

# Khởi tạo embedding nhỏ ngẫu nhiên
user_embeddings = [
    [random.uniform(-0.05, 0.05) for _ in range(EMBEDDING_SIZE)]
    for _ in range(NUM_USERS)
]
movie_embeddings = [
    [random.uniform(-0.05, 0.05) for _ in range(EMBEDDING_SIZE)]
    for _ in range(NUM_MOVIES)
]

# Bias giúp mô hình điều chỉnh theo xu hướng chung của user/phim
user_bias = [0.0] * NUM_USERS
movie_bias = [0.0] * NUM_MOVIES
global_bias = 0.0

for epoch in range(EPOCHS):
    random.shuffle(train_data)
    total_loss = 0.0

    for user_id, movie_id, label in train_data:
        user_vec = user_embeddings[user_id]
        movie_vec = movie_embeddings[movie_id]

        score = global_bias + user_bias[user_id] + movie_bias[movie_id]
        score += sum(u * m for u, m in zip(user_vec, movie_vec))
        prediction = sigmoid(score)

        # Binary cross-entropy loss
        prediction_safe = min(max(prediction, 1e-12), 1.0 - 1e-12)
        total_loss -= (
            label * math.log(prediction_safe)
            + (1 - label) * math.log(1.0 - prediction_safe)
        )

        error = prediction - label

        # Lưu embedding cũ để cập nhật đồng bộ hai vector
        old_user_vec = user_vec[:]
        old_movie_vec = movie_vec[:]

        for k in range(EMBEDDING_SIZE):
            user_vec[k] -= LEARNING_RATE * error * old_movie_vec[k]
            movie_vec[k] -= LEARNING_RATE * error * old_user_vec[k]

        user_bias[user_id] -= LEARNING_RATE * error
        movie_bias[movie_id] -= LEARNING_RATE * error
        global_bias -= LEARNING_RATE * error

    # Đánh giá accuracy trên test
    correct = 0
    for user_id, movie_id, label in test_data:
        score = global_bias + user_bias[user_id] + movie_bias[movie_id]
        score += sum(
            u * m
            for u, m in zip(user_embeddings[user_id], movie_embeddings[movie_id])
        )
        predicted_label = 1 if sigmoid(score) >= 0.5 else 0
        if predicted_label == label:
            correct += 1

    avg_loss = total_loss / len(train_data)
    accuracy = correct / len(test_data)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train loss: {avg_loss:.4f} | "
        f"Test accuracy: {accuracy:.4f}"
    )
USER_ID = 10
TOP_K = 10

# Lấy phim đã xuất hiện trong train cho user này
watched = {
    movie_id
    for user_id, movie_id, _ in train_data
    if user_id == USER_ID
}

recommendations = []

for movie_id in range(NUM_MOVIES):
    if movie_id in watched:
        continue

    score = global_bias + user_bias[USER_ID] + movie_bias[movie_id]
    score += sum(
        u * m
        for u, m in zip(
            user_embeddings[USER_ID],
            movie_embeddings[movie_id]
        )
    )

    recommendations.append((score, movie_id))

recommendations.sort(reverse=True)

print(f"\nTop {TOP_K} phim gợi ý cho user {USER_ID}:")
for rank, (score, movie_id) in enumerate(recommendations[:TOP_K], start=1):
    print(f"{rank}. movie_id đã mã hóa: {movie_id} | điểm mô hình: {score:.4f}")


movie_map_path = Path(r"D:\DOANHOCSAU\movie_map.csv")
movie_titles = {}

with movie_map_path.open("r", newline="", encoding="utf-8") as file:
    for row in csv.DictReader(file):
        movie_titles[int(row["encoded_movie_id"])] = row["title"]

print(f"\nTop {TOP_K} phim có tên cho user {USER_ID}:")
for rank, (score, movie_id) in enumerate(recommendations[:TOP_K], start=1):
    title = movie_titles.get(movie_id, "Không tìm thấy tên")
    print(f"{rank}. {title} | điểm mô hình: {score:.4f}")