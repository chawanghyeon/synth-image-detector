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
- **학습 데이터:** 약 10만 장입니다(28차 v4).
  - OpenFake: 생성기 90여 종의 AI 이미지와 실제 사진, 비공개 상용 생성기(GPT-Image, Midjourney, Imagen 등) 1만 8천 장 추가
  - Codex로 만든 GPT-Image 23장 (5배로 반복)
  - Reddit에 실제로 올라온 AI·실제 이미지 2만 장 (3배로 반복해 더 자주 보게 함)
  - 실제 이미지: COCO, WikiArt 회화, ImageNet-R 그림, CC3M 웹 이미지
  - 모든 이미지는 JPEG 사본과 SNS식 사본(축소 + JPEG)으로도 학습합니다.
- **실행 환경:** torch 없이 onnxruntime으로 돌아갑니다. Apple M3 Pro CPU에서 요청 하나에 약 0.8초, 메모리 약 1.6GB를 씁니다.

이전 버전들의 과정과 실패한 시도는 [docs/experiments.md](docs/experiments.md)에 있습니다.
- 노이즈 통계 SPAM + NoiseNet + DINOv3: 압축된 이미지를 잡지 못했습니다.
- 첫 증류 모델(23차): Reddit 실제 이미지 오탐이 8%였습니다.
- v3r(27차): SNS식으로 처리된 GPT-Image를 절반만 잡았습니다.

## 성능

**판정 기준선(`ai_score > 0.5`)** 은 학습에 쓰지 않은 Reddit 실제 이미지(Reddit 3)에서 오탐 1%가 되도록 정했습니다. 실제 사진을 AI로 잘못 판정하는 것을 더 무겁게 본 설정입니다.

봉인 평가 세트(28차)로 잰 결과입니다. 원본 → SNS식(0.75배 축소 + JPEG 80) 순서입니다.

| 평가 세트 | AI 탐지 | 실제 오탐 |
| --- | --- | --- |
| Reddit 4: 실제 게시물, 학습에 쓰지 않음 (AI 5,798 / 실제 1,417) | 98.2% → 96.4% | 1.3% → 1.3% |
| 새 생성기: OpenFake test 1–2 (AI 6,974 / 실제 6,711) | 95.6% → 88.7% | 0.1% → 0.1% |
| 어려운 실제 이미지: CC3M·WikiArt·ImageNet-R (실제 8,939) | — | 0.0% → 0.1% |
| Codex로 직접 만든 GPT-Image 50장 (학습 23장과 별개) | 50/50 → 39/50 | — |
| OpenFake test 실제 사진 100장 | — | 0/100 → 0/100 |

같은 기준선에서 이전 서버 v3r(27차)은 새 생성기를 89.2% → 79.3%, Codex를 45/50 → 25/50 잡았습니다. Reddit 4와 오탐은 거의 같았습니다.

기준선을 낮추면 탐지가 늘고 오탐도 늘어납니다. 예를 들어 학습 보류 실제 이미지 기준(logit −0.17)으로 낮추면 Reddit 4 탐지가 99.1%로 오르지만 오탐이 3.3%가 됩니다.

## 실행

```sh
UV_PYTHON_INSTALL_DIR=.python uv sync --extra dev
uv run uvicorn detector.serving.app:app --app-dir src --port 8000
```

서버가 시작하려면 `models/`에 `student.onnx`(약 1.3GB)와 `student.json`(입력 크기, 경계값)이 있어야 합니다. 모델은 Git에 넣지 않으며, 아래 "재학습"으로 만듭니다. 이미지는 한 변이 256px 이상이어야 하고, 그보다 작으면 422를 반환합니다. 업로드는 최대 20MiB입니다.

## 한계

- **Reddit 평가의 분포가 학습과 같습니다.** Reddit 3·4는 학습에 쓴 Reddit 0–2와 같은 방식으로 모은 데이터입니다. 다른 SNS나 다른 시기의 이미지에서는 따로 확인하지 않았습니다.
- **SNS식으로 처리된 GPT-Image 이미지 탐지는 39/50입니다.** Codex 평가 이미지는 학습한 23장과 같은 도구로 만들었습니다. 그래서 학습에 없던 새 생성기에서는 이보다 낮을 수 있습니다.
- **측정하지 않은 것:** 스크린샷, 신경망 처리를 거친 최신 폰 사진, 편집 사진의 오탐입니다.
- **라이선스:**
  - 학습 데이터 OpenFake는 CC BY-NC 4.0(비상업)입니다. 그래서 상업 서비스에는 다른 데이터로 다시 학습해야 합니다.
  - PE-Core는 Apache 2.0입니다.

## 구조

```
src/detector/
  preprocess.py      이미지 -> 모델 입력 (CropPolicy: 어느 영역을 볼지, to_array)
  model_card.py      models/student.json (입력 크기, 경계값)
  serving/           FastAPI 서버와 ONNX Detector (torch 없음)
  data/              출처(Source), 원본 해상도 조각 공장(CropFactory), meta.csv, JPEG/SNS 열화
  training/          Recipe, Student(PE-Core + 선형 헤드), Trainer, Kaggle 학습 작업, ONNX 변환
  evaluation/        점수 파일(Scores)의 경계값과 지표, Codex 벤치마크
  kaggle_jobs.py     Kaggle 작업 묶기·실행·대기·다운로드
  cli.py             `detector` 명령
kaggle/<작업>/       Kaggle 노트북 하나 = 설정(main.py) + kernel-metadata.json
tools/               Codex 이미지 생성 스크립트와 프롬프트
data/                (Git 제외) codex_test, codex_train, real_photos, exp28(서버 모델의 학습 결과)
```

**평가 원칙:** 경계값은 학습에 쓰지 않은 세트(Reddit 3) 실제 이미지의 오탐 1% 지점으로 먼저 고정합니다. 그 뒤 다른 세트(Reddit 4, 새 생성기, 어려운 실제 이미지, Codex)는 원본과 SNS 사본 둘 다로 잽니다.

## 재학습

GPU 작업은 Kaggle(T4 2장)에서, 평가와 변환은 노트북에서 합니다. Kaggle 명령줄 도구가 로그인돼 있어야 합니다(`kaggle auth login`).

```sh
UV_PYTHON_INSTALL_DIR=.python uv sync --all-extras
# 1. 데이터: Kaggle CPU 작업. 파트마다 노트북 하나 (detector.data.sources.PARTS)
#    train_a: OpenFake train 12 shard / train_b: COCO, WikiArt, ImageNet-R, CC3M / train_c: Reddit 0-2
#    train_d: 쓰지 않은 OpenFake train shard 40개의 비공개 상용 생성기 AI 이미지
#    eval: OpenFake test 1-2, Reddit 3-4, 어려운 실제 이미지 (학습에 쓰지 않음)
for part in train_a train_b train_c train_d eval; do
  uv run detector push kaggle/data_factory --set "PART=\"$part\"" --suffix ${part//_/-}
done
# Codex GPT-Image 학습 이미지 -> Kaggle 데이터셋 synth-codex-train
tools/codex_images.sh tools/prompts/codex_train.txt data/codex_train
uv run detector pack-codex data/codex_train data/codex_train_ds
kaggle datasets version -p data/codex_train_ds -m "Codex images"
# 2. 학습과 봉인 세트 채점: kaggle/train/main.py의 RECIPES (GPU당 하나, 약 3.5시간)
uv run detector push kaggle/train
uv run detector wait chawanghyeon/train-v2 data/exp31   # 끝나면 내려받고 봉인 세트 결과 출력
uv run detector bench data/exp31 v4                      # Codex 50장 / 실제 사진 100장
# 3. ONNX 변환. 경계값은 Reddit 3 실제 이미지 오탐 1% -> models/student.onnx, student.json
uv run detector export data/exp31/student_v4.pt data/exp31/scores_v4.npz
uv run detector bench --served
```

## 개발

```sh
uv run pre-commit install   # 커밋 때 ruff(lint, format), 파일 검사, pytest 실행
uv run pytest
```

실험 과정과 기각한 방법은 [docs/experiments.md](docs/experiments.md)에 있습니다. 이전 프로젝트의 시도는 [docs/past_attempts.md](docs/past_attempts.md)에 있습니다.
