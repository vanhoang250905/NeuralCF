from pathlib import Path
import csv
import sqlite3

import torch
from flask import Flask, redirect, render_template_string, request, url_for

DATA_DIR = Path(r"D:\DOANHOCSAU")
TRAIN_FILE = DATA_DIR / "train.csv"
MOVIE_MAP_FILE = DATA_DIR / "movie_map.csv"
DB_FILE = DATA_DIR / "feedback.db"

TOP_K = 10
SHRINKAGE = 10.0

app = Flask(__name__)


def load_csv(path):
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


train_rows = load_csv(TRAIN_FILE)
movie_rows = load_csv(MOVIE_MAP_FILE)

NUM_USERS = max(int(row["user_id"]) for row in train_rows) + 1
NUM_MOVIES = max(int(row["encoded_movie_id"]) for row in movie_rows) + 1

movie_titles = {
    int(row["encoded_movie_id"]): row["title"]
    for row in movie_rows
}

# Hồ sơ tương tác train cho từng user
seen_by_user = [set() for _ in range(NUM_USERS)]
liked_by_user = [set() for _ in range(NUM_USERS)]
movie_popularity = [0] * NUM_MOVIES

# Ma trận phim × user: 1 nếu user thích phim
item_user = torch.zeros((NUM_MOVIES, NUM_USERS), dtype=torch.float32)

for row in train_rows:
    user_id = int(row["user_id"])
    movie_id = int(row["movie_id"])
    label = int(row["label"])

    seen_by_user[user_id].add(movie_id)

    if label == 1:
        liked_by_user[user_id].add(movie_id)
        movie_popularity[movie_id] += 1
        item_user[movie_id, user_id] = 1.0

# Tính độ tương tự item-item
co_likes = item_user @ item_user.T
norms = torch.sqrt(torch.diag(co_likes)).clamp(min=1e-8)
similarity = co_likes / (norms[:, None] * norms[None, :])
similarity *= co_likes / (co_likes + SHRINKAGE)
similarity.fill_diagonal_(0.0)


def init_database():
    with sqlite3.connect(DB_FILE) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                label INTEGER NOT NULL,
                PRIMARY KEY (user_id, movie_id)
            )
        """)


def get_recommendations(user_id):
    with sqlite3.connect(DB_FILE) as connection:
        feedback_rows = connection.execute(
            "SELECT movie_id, label FROM feedback WHERE user_id = ?",
            (user_id,),
        ).fetchall()

    feedback = {int(movie_id): int(label) for movie_id, label in feedback_rows}

    liked_movies = set(liked_by_user[user_id])
    for movie_id, label in feedback.items():
        if label == 1:
            liked_movies.add(movie_id)
        else:
            liked_movies.discard(movie_id)

    seen_movies = set(seen_by_user[user_id]) | set(feedback)

    candidates = [
        movie_id for movie_id in range(NUM_MOVIES)
        if movie_id not in seen_movies
    ]

    if liked_movies:
        profile = torch.zeros(NUM_MOVIES, dtype=torch.float32)
        profile[list(liked_movies)] = 1.0
        scores = (profile @ similarity).tolist()
    else:
        # User chưa có lịch sử thích: tạm gợi ý phim phổ biến
        scores = [float(count) for count in movie_popularity]

    ranked = sorted(
        candidates,
        key=lambda movie_id: scores[movie_id],
        reverse=True,
    )[:TOP_K]

    return [
        {
            "movie_id": movie_id,
            "title": movie_titles.get(movie_id, "Unknown"),
            "score": scores[movie_id],
        }
        for movie_id in ranked
    ]


PAGE = """
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CineMatch — Gợi ý phim</title>
  <style>
    * { box-sizing: border-box; }

    :root {
      --bg: #090b10;
      --panel: #11151e;
      --panel-light: #171c27;
      --text: #f5f6fa;
      --muted: #9299aa;
      --accent: #b7f36b;
      --line: rgba(255,255,255,.09);
    }

    body {
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(ellipse at 15% 0%, #26301d 0, transparent 35%),
        radial-gradient(ellipse at 90% 12%, #25203c 0, transparent 32%),
        var(--bg);
      font-family: Inter, "Segoe UI", Arial, sans-serif;
      min-height: 100vh;
    }

    .container { width: min(1160px, calc(100% - 40px)); margin: auto; }

    header {
      height: 76px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--line);
    }

    .brand { display: flex; align-items: center; gap: 11px; font-weight: 800; }
    .brand-icon {
      width: 38px; height: 38px; display: grid; place-items: center;
      border-radius: 12px; background: var(--accent); color: #111;
      font-size: 20px;
    }
    .brand-name { font-size: 18px; letter-spacing: -.4px; }
    .brand-name span { color: var(--accent); }

    .header-note { color: var(--muted); font-size: 13px; }

    .hero {
      display: grid;
      grid-template-columns: 1.4fr .8fr;
      align-items: center;
      gap: 24px;
      padding: 52px 0 38px;
    }

    .eyebrow {
      color: var(--accent); text-transform: uppercase; letter-spacing: 2px;
      font-size: 11px; font-weight: 800;
    }

    h1 {
      margin: 12px 0;
      max-width: 650px;
      font-size: clamp(34px, 5vw, 58px);
      line-height: 1.04;
      letter-spacing: -2.5px;
    }

    h1 span { color: var(--accent); }
    .intro { max-width: 570px; color: var(--muted); line-height: 1.7; }

    .user-panel {
      padding: 21px;
      background: linear-gradient(145deg, #171c27, #10131b);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 16px 50px #0005;
    }

    .user-panel label {
      display: block; color: var(--muted); font-size: 12px;
      margin-bottom: 10px;
    }

    .user-controls { display: flex; gap: 9px; }

    select, button {
      font: inherit; border: 0; border-radius: 10px;
    }

    select {
      flex: 1; min-width: 0; padding: 12px;
      color: var(--text); background: #222836;
    }

    button { cursor: pointer; transition: .18s ease; }
    button:hover { transform: translateY(-1px); filter: brightness(1.08); }

    .primary {
      padding: 12px 16px; color: #111; font-weight: 800;
      background: var(--accent);
    }

    .stats {
      display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px;
    }

    .stat {
      padding: 8px 11px; border-radius: 999px;
      color: #c4c9d4; background: #ffffff0b; font-size: 11px;
      border: 1px solid var(--line);
    }

    .section-heading {
      display: flex; justify-content: space-between; align-items: end;
      gap: 16px; margin: 18px 0;
    }

    h2 { margin: 0; font-size: 24px; letter-spacing: -.6px; }
    .section-sub { color: var(--muted); margin-top: 7px; font-size: 13px; }
    .count { color: var(--accent); font-size: 13px; font-weight: 700; }

    .grid {
      display: grid; grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 16px; padding-bottom: 42px;
    }

    .movie-card {
      overflow: hidden; border: 1px solid var(--line); border-radius: 15px;
      background: var(--panel); transition: .2s ease;
    }

    .movie-card:hover {
      transform: translateY(-5px); border-color: #b7f36b66;
      box-shadow: 0 18px 38px #0006;
    }

    .poster {
      height: 178px; position: relative; display: flex;
      align-items: end; padding: 15px; overflow: hidden;
      background:
        radial-gradient(circle at 75% 25%, #ffffff45, transparent 20%),
        linear-gradient(145deg, #564c83, #211d38 70%);
    }
    .poster::before {
      content: ""; position: absolute; width: 145px; height: 145px;
      right: -32px; top: -38px; border: 1px solid #ffffff30;
      border-radius: 50%; box-shadow: 0 0 0 18px #ffffff08, 0 0 0 38px #ffffff06;
    }
    .poster-1 { background: linear-gradient(145deg, #76523d, #291d26 75%); }
    .poster-2 { background: linear-gradient(145deg, #375b69, #172632 75%); }
    .poster-3 { background: linear-gradient(145deg, #655d31, #242319 75%); }
    .poster-4 { background: linear-gradient(145deg, #723e50, #291824 75%); }
    .poster-5 { background: linear-gradient(145deg, #345b45, #17271f 75%); }

    .rank {
      position: absolute; top: 12px; left: 12px;
      padding: 6px 9px; border-radius: 8px;
      color: #10130d; background: var(--accent);
      font-size: 11px; font-weight: 900;
    }

    .poster-symbol {
      position: relative; z-index: 1; font-size: 39px;
      filter: drop-shadow(0 4px 10px #0008);
    }

    .movie-info { padding: 13px; }
    .movie-title {
      min-height: 42px; font-size: 14px; line-height: 1.45;
      font-weight: 700;
    }
    .score {
      color: var(--muted); font-size: 11px; margin-top: 7px;
    }

    .actions { display: flex; gap: 7px; margin-top: 13px; }
    .actions form { flex: 1; }
    .actions button {
      width: 100%; padding: 9px 5px; color: #dfe3ec;
      background: #ffffff0b; border: 1px solid var(--line);
      font-size: 11px;
    }
    .actions .like:hover { color: var(--accent); }
    .actions .dislike:hover { color: #ff8989; }

    .empty {
      padding: 38px; text-align: center; color: var(--muted);
      border: 1px dashed var(--line); border-radius: 15px;
    }

    footer {
      padding: 20px 0 30px; border-top: 1px solid var(--line);
      color: var(--muted); font-size: 12px;
    }

    @media (max-width: 950px) {
      .grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    }
    @media (max-width: 650px) {
      .container { width: min(100% - 26px, 520px); }
      .header-note { display: none; }
      .hero { grid-template-columns: 1fr; padding-top: 34px; }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 11px; }
      .poster { height: 150px; }
      h1 { letter-spacing: -1.5px; }
    }
    @media (max-width: 370px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">
        <div class="brand-icon">▶</div>
        <div class="brand-name">Cine<span>Match</span></div>
      </div>
      <div class="header-note">Khám phá bộ phim tiếp theo của bạn</div>
    </header>

    <section class="hero">
      <div>
        <div class="eyebrow">MovieLens · Hệ thống gợi ý phim</div>
        <h1>Phim hay dành riêng <span>cho bạn.</span></h1>
        <p class="intro">
          Khám phá những bộ phim được đề xuất từ lịch sử đánh giá.
          Đánh dấu thích hoặc không thích để cập nhật danh sách lần sau.
        </p>
        <div class="stats">
          <span class="stat">🎞 1.682 phim</span>
          <span class="stat">👥 943 hồ sơ</span>
          <span class="stat">✨ Item-Item CF</span>
        </div>
      </div>

      <form class="user-panel" method="get" action="/">
        <label for="user_id">ĐANG XEM GỢI Ý CHO</label>
        <div class="user-controls">
          <select id="user_id" name="user_id">
            {% for uid in user_ids %}
              <option value="{{ uid }}" {% if uid == user_id %}selected{% endif %}>
                Hồ sơ người dùng {{ uid }}
              </option>
            {% endfor %}
          </select>
          <button class="primary" type="submit">Xem phim</button>
        </div>
      </form>
    </section>

    <section>
      <div class="section-heading">
        <div>
          <h2>Đề xuất cho bạn</h2>
          <div class="section-sub">Dựa trên các phim có sở thích tương tự</div>
        </div>
        <div class="count">TOP {{ recommendations|length }}</div>
      </div>

      {% if recommendations %}
        <div class="grid">
          {% for movie in recommendations %}
            <article class="movie-card">
              <div class="poster poster-{{ loop.index0 % 6 }}">
                <span class="rank">#{{ loop.index }}</span>
                <span class="poster-symbol">🎬</span>
              </div>
              <div class="movie-info">
                <div class="movie-title">{{ movie.title }}</div>
                <div class="score">
                  Điểm phù hợp: {{ "%.3f"|format(movie.score) }}
                </div>
                <div class="actions">
                  <form method="post" action="/feedback">
                    <input type="hidden" name="user_id" value="{{ user_id }}">
                    <input type="hidden" name="movie_id" value="{{ movie.movie_id }}">
                    <input type="hidden" name="label" value="1">
                    <button class="like" type="submit">👍 Thích</button>
                  </form>
                  <form method="post" action="/feedback">
                    <input type="hidden" name="user_id" value="{{ user_id }}">
                    <input type="hidden" name="movie_id" value="{{ movie.movie_id }}">
                    <input type="hidden" name="label" value="0">
                    <button class="dislike" type="submit">👎 Bỏ qua</button>
                  </form>
                </div>
              </div>
            </article>
          {% endfor %}
        </div>
      {% else %}
        <div class="empty">Chưa có phim phù hợp để gợi ý cho hồ sơ này.</div>
      {% endif %}
    </section>

    <footer>
      CineMatch · Đồ án hệ thống gợi ý phim · Phản hồi được lưu trên máy này.
    </footer>
  </div>
</body>
</html>
"""
@app.route("/", methods=["GET"])
def home():
    try:
        user_id = int(request.args.get("user_id", "10"))
    except ValueError:
        user_id = 10

    user_id = max(0, min(user_id, NUM_USERS - 1))

    return render_template_string(
        PAGE,
        user_id=user_id,
        user_ids=range(NUM_USERS),
        recommendations=get_recommendations(user_id),
    )


@app.route("/feedback", methods=["POST"])
def save_feedback():
    try:
        user_id = int(request.form["user_id"])
        movie_id = int(request.form["movie_id"])
        label = int(request.form["label"])
    except (KeyError, ValueError):
        return "Phản hồi không hợp lệ", 400

    if not (0 <= user_id < NUM_USERS):
        return "User ID không hợp lệ", 400
    if not (0 <= movie_id < NUM_MOVIES) or label not in (0, 1):
        return "Phim hoặc phản hồi không hợp lệ", 400

    with sqlite3.connect(DB_FILE) as connection:
        connection.execute(
            """
            INSERT INTO feedback (user_id, movie_id, label)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, movie_id)
            DO UPDATE SET label = excluded.label
            """,
            (user_id, movie_id, label),
        )

    return redirect(url_for("home", user_id=user_id))


if __name__ == "__main__":
    init_database()
    print("Mở trình duyệt tại http://127.0.0.1:5000")
    app.run(debug=False)