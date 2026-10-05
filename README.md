# synth-image-detector

이미지를 올리면 AI 생성 이미지일 가능성을 `ai_score`(0–1)로 돌려주는 가벼운 FastAPI 서버입니다.

```sh
curl -F file=@photo.jpg http://127.0.0.1:8000/predict
# {"ai_score": 0.94}
```

**판정 기준:** `ai_score > 0.5`이면 AI 생성으로 봅니다. 0.5는 validation 실제 사진의 오탐률이 1%가 되도록 맞춘 경계입니다. `ai_score`는 **보정된 확률이 아닙니다.** 낮은 점수도 실제 사진이라는 증명이 아닙니다.

## 방법

AI 생성 이미지에 남는 픽셀 수준 노이즈 흔적을 서로 다른 두 방식으로 측정해 합칩니다. 둘 다 이미지를 리사이즈하지 않고 원본 해상도 조각에서 계산합니다.

| 구성 | 원리 | 크기 |
| --- | --- | --- |
| SPAM + gradient boosting | 이웃 픽셀 차분이 이어지는 패턴의 통계(마르코프 전이 확률 1,372개)를 분류. 512px 조각 최대 4개 | 5.2MB |
| NoiseNet (ONNX) | 이미지에서 내용을 빼고 남은 잔차(3×3 평활, 2× 업샘플링)를 작은 CNN(파라미터 58만 개)이 판별. 128px 조각 최대 32개 | 2.3MB |

두 점수는 validation 실제 사진 기준으로 정규화한 뒤 0.7 : 0.3으로 더합니다(`models/ensemble.json`). 서버에는 torch도 GPU도 필요 없고, 의존성은 numpy·pillow·scikit-learn·onnxruntime뿐입니다. Apple M3 Pro CPU에서 이미지당 약 0.3초가 걸립니다.

## 실행

```sh
UV_PYTHON_INSTALL_DIR=.python uv sync --extra dev
uv run uvicorn detector.app:app --app-dir src --port 8000
```

`models/`에 `spam_hgb.joblib`, `noisenet.onnx`, `ensemble.json`이 있어야 서버가 시작됩니다. 모델은 Git에 넣지 않으며 아래 "재학습"으로 만듭니다. 이미지는 한 변이 256px 이상이어야 하고, 그보다 작으면 422를 반환합니다. 업로드는 최대 20MiB입니다.

## 성능 (2026-10-05, 서버와 같은 경로로 측정)

학습 데이터는 [OpenFake v2](https://huggingface.co/datasets/ComplexDataLab/OpenFake) train shard 6개(73개 생성기)와 실제 사진 COCO val2017·Flickr30k 각 약 5천 장입니다. 정규화·가중치·임계값은 OpenFake validation에서 정했습니다. 테스트셋은 OpenFake test 6,872장으로, **학습에 없는 생성기 20종**(GPT-image-1.5/2, Midjourney 7, FLUX.2-klein, Seedream 5, Sora-2, Veo-3 등)과 **학습에 없는 실제 사진 출처**(ImageNet, DOCCI)로만 구성됩니다.

| 테스트 조건 | AUROC | `ai_score > 0.5`일 때 실제 사진 오탐 | AI 이미지 탐지 |
| --- | ---: | ---: | ---: |
| 원본 파일 | **0.964** | **0.23%** | **52.6%** |
| JPEG 90으로 다시 저장 | 0.912 | 0.21% | 18.8% |
| JPEG 75로 다시 저장 | 0.879 | 0.18% | 13.7% |

탐지율은 생성기마다 크게 다릅니다(SPAM 단독, 512px 조각 평가). Midjourney 7은 99%, Ideogram·Lumina는 90% 이상, GPT-image-1.5와 z-image-turbo는 60%대입니다. GPT-image-2와 Sora-2는 20%대, FLUX.2-klein은 14%, Seedream 5는 6%, Recraft는 0–15%입니다.

## 한계

- **재압축된 이미지는 대부분 놓칩니다.** JPEG로 다시 저장하면 순위 정보(AUROC 0.88–0.91)는 남지만, 모든 점수가 낮아져 0.5를 넘는 경우가 크게 줄어듭니다. 메신저·SNS를 거친 이미지는 탐지율이 낮다고 봐야 합니다.
- 실제 사진 테스트셋은 ImageNet·DOCCI뿐입니다. 스크린샷, 보정·편집된 사진, 스캔 이미지의 오탐률은 측정하지 않았습니다.
- 학습 데이터 OpenFake는 CC BY-NC 4.0입니다. 상업 서비스에 쓰기 전에 라이선스를 확인해야 합니다.

## 재학습

```sh
UV_PYTHON_INSTALL_DIR=.python uv sync --all-extras
# OpenFake shard(train 6개, validation·test 각 1개) → 원본 해상도 512px 조각
uv run python scripts/prepare_data.py data/raw/core/*.parquet
# 추가 실제 사진: COCO val2017, Flickr30k에서 각 5,000장
uv run python scripts/prepare_data.py --real coco data/raw/val2017
uv run python scripts/prepare_data.py --real flickr30k data/raw/flickr30k/flickr30k-images
uv run python scripts/train_spam.py       # models/spam_hgb.joblib, reports/spam_hgb.json (CPU 약 1시간)
uv run python scripts/train_noisenet.py   # models/noisenet.onnx, reports/noisenet.json (M3 Pro 약 1.2시간)
uv run python scripts/fit_ensemble.py     # models/ensemble.json, reports/ensemble.json (약 50분)
```

사용한 OpenFake train shard: `train-00000-of-00032-00000`, `-00005-…-00003`, `-00011-…-00005`, `-00016-…-00007`, `-00022-…-00010`, `-00027-…-00012`. validation·test는 각각 첫 shard입니다.

실험 과정과 기각한 방법은 [docs/experiments.md](docs/experiments.md), 이전 프로젝트의 시도는 [docs/past_attempts.md](docs/past_attempts.md)에 있습니다.
