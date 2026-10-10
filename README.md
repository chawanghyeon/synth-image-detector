# synth-image-detector

이미지를 올리면 AI 생성 이미지일 가능성을 `ai_score`(0–1)로 돌려주는 FastAPI 서버입니다.

```sh
curl -F file=@photo.jpg http://127.0.0.1:8000/predict
# {"ai_score": 0.94}
```

**판정 기준:** `ai_score > 0.5`이면 AI 생성으로 봅니다. 0.5는 학습에 쓰지 않은 Reddit 실제 게시물 이미지의 오탐률이 1%가 되도록 맞춘 경계입니다. `ai_score`는 **보정된 확률이 아닙니다.** 낮은 점수도 실제 사진이라는 증명이 아닙니다.

## 방법

[PE-Core-L14](https://huggingface.co/facebook/PE-Core-L14-336)(Meta Perception Encoder, 3억 파라미터)의 마지막 8개 블록을 미세조정한 판정기 하나를 씁니다(`models/student.onnx`).

- **입력:** 이미지 가운데에서 원본 해상도로 최대 512px 정사각형을 자른 뒤 336px로 줄입니다. 리사이즈 전에 자르므로 큰 이미지도 세부가 보존됩니다.
- **학습 데이터:** 약 8만 3천 장입니다(27차 v3r).
  - OpenFake: 생성기 90여 종의 AI 이미지와 실제 사진
  - Reddit에 실제로 올라온 AI·실제 이미지 2만 장 (3배로 반복해 더 자주 보게 함)
  - 실제 이미지: COCO, WikiArt 회화, ImageNet-R 그림, CC3M 웹 이미지
  - 모든 이미지는 JPEG 사본과 SNS식 사본(축소 + JPEG)으로도 학습합니다.
- **실행 환경:** torch 없이 onnxruntime으로 돌아갑니다. Apple M3 Pro CPU에서 요청 하나에 약 0.8초, 메모리 약 1.6GB를 씁니다.

이전 버전들의 과정과 실패한 시도는 [docs/experiments.md](docs/experiments.md)에 있습니다.
- 노이즈 통계 SPAM + NoiseNet + DINOv3: 압축된 이미지를 잡지 못했습니다.
- 첫 증류 모델(23차): Reddit 실제 이미지 오탐이 8%였습니다.

## 성능

**판정 기준선(`ai_score > 0.5`)** 은 학습에 쓰지 않은 Reddit 실제 이미지(Reddit 3)에서 오탐 1%가 되도록 정했습니다. 실제 사진을 AI로 잘못 판정하는 것을 더 무겁게 본 설정입니다.

봉인 평가 세트(27차)로 잰 결과입니다. 원본 → SNS식(0.75배 축소 + JPEG 80) 순서입니다.

| 평가 세트 | AI 탐지 | 실제 오탐 |
| --- | --- | --- |
| Reddit 4: 실제 게시물, 처음 개봉 (AI 5,798 / 실제 1,417) | 98.2% → 96.6% | 1.4% → 1.1% |
| 새 생성기: OpenFake test 1–2 (AI 6,974 / 실제 6,711) | 89.2% → 79.3% | 0.0% → 0.0% |
| 어려운 실제 이미지: CC3M·WikiArt·ImageNet-R (실제 8,939) | — | 0.0% → 0.1% |
| Codex로 직접 만든 GPT-Image 50장 | 45/50 → 25/50 | — |
| OpenFake test 실제 사진 100장 | — | 0/100 → 0/100 |

같은 기준선에서 이전 서버는 Reddit 4를 52.8% → 45.7%, 새 생성기를 81.5% → 65.9%, Codex를 37/50 → 15/50 잡았습니다.

기준선을 낮추면 탐지가 늘고 오탐도 늘어납니다. 예를 들어 학습 보류 실제 이미지 기준(logit −0.31)으로 낮추면 결과가 이렇습니다.
- Reddit 4: 99.4% 탐지, 오탐 3.4%
- 새 생성기: 96.2% 탐지
- Codex: 50/50 → 36/50

## 실행

```sh
UV_PYTHON_INSTALL_DIR=.python uv sync --extra dev
uv run uvicorn detector.app:app --app-dir src --port 8000
```

서버가 시작하려면 `models/`에 `student.onnx`(약 1.3GB)와 `student.json`(입력 크기, 경계값)이 있어야 합니다. 모델은 Git에 넣지 않으며, 아래 "재학습"으로 만듭니다. 이미지는 한 변이 256px 이상이어야 하고, 그보다 작으면 422를 반환합니다. 업로드는 최대 20MiB입니다.

## 한계

- **Reddit 평가의 분포가 학습과 같습니다.** Reddit 3·4는 학습에 쓴 Reddit 0–2와 같은 방식으로 모은 데이터입니다. 다른 SNS나 다른 시기의 이미지에서는 따로 확인하지 않았습니다.
- **SNS식으로 처리된 GPT-Image 이미지는 절반만 잡습니다.** 이 기준선에서 Codex 이미지의 SNS 사본 탐지가 25/50입니다.
- **측정하지 않은 것:** 스크린샷, 신경망 처리를 거친 최신 폰 사진, 편집 사진의 오탐입니다.
- **라이선스:**
  - 학습 데이터 OpenFake는 CC BY-NC 4.0(비상업)입니다. 그래서 상업 서비스에는 다른 데이터로 다시 학습해야 합니다.
  - PE-Core는 Apache 2.0입니다.

## 재학습

GPU 작업은 Kaggle(T4 2장)에서, 평가와 변환은 노트북에서 합니다. Kaggle 명령줄 도구가 로그인돼 있어야 합니다(`kaggle auth login`).

```sh
UV_PYTHON_INSTALL_DIR=.python uv sync --all-extras
# 1. 데이터: Kaggle CPU 작업 4개. 원본 해상도 512px 조각을 *.bin(tar)으로 묶어 노트북 출력으로 남긴다.
#    train_a: OpenFake train 12 shard / train_b: COCO, WikiArt, ImageNet-R, CC3M / train_c: Reddit 0-2
#    eval: OpenFake test 1-2, Reddit 3-4, 어려운 실제 이미지 (학습에 쓰지 않음)
for part in train_a train_b train_c eval; do
  uv run python kaggle/push.py kaggle/data_factory --set "PART=\"$part\"" --suffix ${part//_/-}
done
# 2. 학습과 봉인 세트 채점 (VARIANTS의 v3r, GPU당 약 3시간)
uv run python kaggle/push.py kaggle/train
kaggle kernels output chawanghyeon/train-v2 -p data/exp27
uv run python scripts/eval_sealed.py data/exp27
# 3. ONNX 변환. 기준선은 Reddit 3 실제 이미지 오탐 1% -> models/student.onnx, models/student.json
uv run python scripts/export_student.py data/exp27/student_v3r.pt data/exp27/scores_v3r.npz reddit3
```

이전 방식(교사 PE-Core-G의 증류, 23차)은 `kaggle/siglip_features`, `scripts/make_teacher.py`, `kaggle/distill`로 다시 만들 수 있습니다.

실험 과정과 기각한 방법은 [docs/experiments.md](docs/experiments.md)에 있습니다. 이전 프로젝트의 시도는 [docs/past_attempts.md](docs/past_attempts.md)에, 초기 조사 계획은 [docs/plan.md](docs/plan.md)에 있습니다.
