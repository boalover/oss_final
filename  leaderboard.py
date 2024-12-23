# leaderboard.py
import json

# 예시: 리더보드를 저장할 파일 경로
LEADERBOARD_FILE = "leaderboard.json"

def load_leaderboard():
    """리더보드를 파일에서 로드."""
    try:
        with open(LEADERBOARD_FILE, "r") as f:
            leaderboard = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        leaderboard = {}
    return leaderboard

def save_leaderboard(leaderboard):
    """리더보드를 파일에 저장."""
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(leaderboard, f, indent=4)

def update_leaderboard(user, score):
    """리더보드 업데이트."""
    leaderboard = load_leaderboard()
    leaderboard[user] = score
    save_leaderboard(leaderboard)

def display_leaderboard():
    """리더보드 표시."""
    leaderboard = load_leaderboard()
    sorted_leaderboard = sorted(leaderboard.items(), key=lambda x: x[1], reverse=True)
    
    print("Leaderboard:")
    for user, score in sorted_leaderboard:
        print(f"{user}: {score}")
