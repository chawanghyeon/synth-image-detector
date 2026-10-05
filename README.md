# synth-image-detector

이미지를 올리면 AI 생성 이미지일 가능성을 `ai_score`(0–1)로 돌려주는 가벼운 FastAPI 서버입니다.

```sh
curl -F file=@photo.jpg http://127.0.0.1:8000/predict
# {"ai_score": 0.94}
```

**판정 기준:** `ai_score > 0.5`이면 AI 생성으로 봅니다. 0.5는 validation 실제 사진의 오탐률이 1%가 되도록 맞춘 경계입니다. `ai_score`는 **보정된 확률이 아닙니다.** 낮은 점수도 실제 사진이라는 증명이 아닙니다.

## 방법

서로 다른 성격의 탐지기 세 개를 합칩니다. 모두 이미지를 리사이즈하지 않고 원본 해상도 조각에서 계산합니다.

| 구성 | 원리 | 강한 조건 | 크기 |
| --- | --- | --- | --- |
| SPAM + gradient boosting | 이웃 픽셀 차분이 이어지는 패턴의 통계(마르코프 전이 확률 1,372개). 512px 조각 최대 4개 | 원본 파일 | 5.2MB |
| NoiseNet (ONNX) | 이미지 내용을 빼고 남은 노이즈 잔차를 작은 CNN(파라미터 58만 개)이 판별. 128px 조각 최대 32개 | 원본, 약한 압축 | 2.3MB |
| 의미 탐지기 (ONNX) | 고정된 [DINOv3 ViT-B/16](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)의 세 깊이 특징(CLS·패치 평균·패치 표준편차) + 로지스틱 회귀. 압축·축소 사본으로 함께 학습. 224px 조각 최대 4개 | 재압축, 축소, SNS 처리 | 343MB |

노이즈 탐지기 두 개(0.7 : 0.3)와 의미 탐지기는 각각 validation 실제 사진 기준으로 정규화한 뒤 더합니다(`models/ensemble.json`). 서버에는 torch도 GPU도 필요 없고, 의존성은 numpy·pillow·scikit-learn·onnxruntime뿐입니다. Apple M3 Pro CPU에서 3MP 이미지 하나에 약 0.7초가 걸리고, 최대 메모리는 약 1.1GB입니다.

## 실행

```sh
UV_PYTHON_INSTALL_DIR=.python uv sync --extra dev
uv run uvicorn detector.app:app --app-dir src --port 8000
```

`models/`에 `spam_hgb.joblib`, `noisenet.onnx`, `semantic.onnx`, `semantic_lr.joblib`, `ensemble.json`이 있어야 서버가 시작됩니다. 모델은 Git에 넣지 않으며 아래 "재학습"으로 만듭니다. 이미지는 한 변이 256px 이상이어야 하고, 그보다 작으면 422를 반환합니다. 업로드는 최대 20MiB입니다.

## 성능 (2026-10-06, 서버 그대로 측정)

학습 데이터는 [OpenFake v2](https://huggingface.co/datasets/ComplexDataLab/OpenFake) train shard 6개(73개 생성기)와 실제 사진 COCO val2017·Flickr30k 각 약 5천 장입니다. 정규화와 임계값은 OpenFake validation에서 정했습니다.

평가는 개발·선정에 한 번도 쓰지 않은 새 데이터로 했습니다(`reports/shipped_eval.json`). 하나는 OpenFake test shard 4로, **학습에 없는 생성기**와 **학습에 없는 실제 사진 출처**로 구성됩니다. 다른 하나는 Reddit에 실제로 올라온 이미지 세트 shard 3입니다. 각각 1,200장 안팎입니다. 오탐과 탐지는 `ai_score > 0.5` 기준입니다.

| 테스트 조건 | OpenFake AUROC | 실제 사진 오탐 | AI 탐지 | Reddit AUROC | 실제 사진 오탐 | AI 탐지 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 원본 파일 | 0.953 | 0.9% | **69.1%** | 0.951 | 1.2% | **60.5%** |
| JPEG 75로 다시 저장 | 0.910 | 0.7% | 46.4% | 0.918 | 0.0% | 36.0% |
| SNS식(긴 변 1080px + JPEG 80) | 0.914 | 0.7% | 48.8% | 0.861 | **3.7%** | 48.5% |

노이즈 탐지기 두 개만 쓰던 이전 서버의 Reddit 원본 결과는 오탐 1.0%, 탐지 34.7%였습니다. 같은 조건에서 탐지가 60.5%로 올랐습니다. 탐지기별 비교와 모든 실험 기록은 [docs/experiments.md](docs/experiments.md)에 있습니다.

## 한계

- **Reddit 이미지를 SNS식으로 다시 처리하면 실제 사진 오탐이 3.7%로 올라갑니다.** 임계값을 원본 기준(오탐 1%)으로 정했기 때문입니다. 서비스 입력이 대부분 SNS를 거친 이미지라면 그 분포로 임계값을 다시 정해야 합니다.
- 생성기마다 탐지율이 다릅니다. FLUX.2-klein, Seedream 5, Recraft 계열은 노이즈 탐지기만으로는 15% 이하로 잡혔습니다.
- 실제 사진 테스트는 ImageNet, DOCCI, Reddit 이미지뿐입니다. 스크린샷, 보정·편집 사진, 스캔 이미지의 오탐률은 측정하지 않았습니다.
- **라이선스.** 학습 데이터 OpenFake는 CC BY-NC 4.0(비상업)이므로, 상업 서비스에는 다른 데이터로 다시 학습해야 합니다. DINOv3는 [DINOv3 License](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m/blob/main/LICENSE.md)로 상업적 사용이 가능합니다. 다만 출처 표기, 군사·무기 용도 금지, 수출 규제 준수, 재배포 시 라이선스 사본 첨부라는 조건이 붙습니다. DINOv3는 Hugging Face에서 접근 승인을 받아야 합니다.

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
uv run python scripts/train_semantic.py   # models/semantic.onnx, models/semantic_lr.joblib (약 10분)
uv run python scripts/fit_ensemble.py     # models/ensemble.json, reports/ensemble.json (validation shard 필요, 약 15분)
```

사용한 OpenFake train shard: `train-00000-of-00032-00000`, `-00005-…-00003`, `-00011-…-00005`, `-00016-…-00007`, `-00022-…-00010`, `-00027-…-00012`. validation·test는 각각 첫 shard입니다.

실험 과정과 기각한 방법은 [docs/experiments.md](docs/experiments.md), 이전 프로젝트의 시도는 [docs/past_attempts.md](docs/past_attempts.md)에 있습니다.
