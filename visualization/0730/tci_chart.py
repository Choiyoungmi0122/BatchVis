import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'

# 등급을 수치로 변환
level_map = {"하": 1, "중": 2, "상": 3}

# JSON 로드
with open("C:/Users/Young-Mi Choi/Desktop/dev/batchpro/myapi/responses/experiment_20250724_154245.json", "r", encoding="utf-8") as f:
    data = json.load(f)

records = data["history"]

# 환자별 TCI 정보 추출
plot_data = []
labels = []
for i, record in enumerate(records):
    detail = record["prompts"][0]["detail"]
    temperament = detail["temperament"]
    character = detail["character"]

    profile = {
        **{k: level_map[v] for k, v in temperament.items()},
        **{k: level_map[v] for k, v in character.items()}
    }
    plot_data.append(list(profile.values()))
    labels.append(f"Sim {i+1}")

# 항목 이름 정렬 고정
traits = list(temperament.keys()) + list(character.keys())
N = len(traits)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
plot_data = [d + [d[0]] for d in plot_data]  # 원형 연결
angles += angles[:1]

# 플롯
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
for i, d in enumerate(plot_data):
    ax.plot(angles, d, label=labels[i])
    ax.fill(angles, d, alpha=0.1)

ax.set_thetagrids(np.degrees(angles[:-1]), traits)
plt.title("TCI 기질 + 성격 비교 (Radar Chart)")
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
plt.tight_layout()
plt.show()
