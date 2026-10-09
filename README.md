# synth-image-detector

이미지를 올리면 AI 생성 이미지일 가능성을 `ai_score`(0–1)로 돌려주는 FastAPI 서버입니다.

```sh
curl -F file=@photo.jpg http://127.0.0.1:8000/predict
# {"ai_score": 0.94}
```

**판정 기준:** `ai_score > 0.5`이면 AI 생성으로 봅니다. 0.5는 학습에서 따로 떼어 둔 실제 이미지의 오탐률이 1%가 되도록 맞춘 경계입니다. `ai_score`는 **보정된 확률이 아닙니다.** 낮은 점수도 실제 사진이라는 증명이 아닙니다.

## 방법

[PE-Core-L14](https://huggingface.co/facebook/PE-Core-L14-336)(Meta Perception Encoder, 3억 파라미터)를 미세조정한 판정기 하나를 씁니다(`models/student.onnx`).

- **증류:** 훨씬 큰 PE-Core-G14(19억)에 선형 판정기를 얹은 교사의 판단을 따라 학습했습니다(`kaggle/distill`).
- **입력:** 이미지 가운데에서 원본 해상도로 최대 512px 정사각형을 잘라 336px로 줄입니다. 리사이즈 전에 자르므로 큰 이미지도 세부가 보존됩니다.
- **압축에 강한 이유:** 학습 때 같은 이미지의 JPEG 사본과 SNS식 사본(축소 + JPEG)을 함께 보여 줬습니다. 그래서 SNS에 올라갔다 내려온 이미지에서도 판정이 무너지지 않습니다.
- **실행 환경:** torch 없이 onnxruntime으로 돌아갑니다. Apple M3 Pro CPU에서 요청 하나에 약 0.8초, 메모리 약 1.6GB를 씁니다.

이전 버전(노이즈 통계 SPAM + NoiseNet + DINOv3 결합)은 SNS식으로 처리된 이미지에서 거의 잡지 못해서 교체했습니다. 그 과정과 실패한 시도는 [docs/experiments.md](docs/experiments.md)에 있습니다.

## 성능

### 처음 보는 생성기 (2026-10-09)

[docs/experiments.md](docs/experiments.md) 23차 실험입니다.

- **AI 이미지:** OpenFake test의 2,300장입니다. 학습에 없는 생성기 20종으로 만들어졌습니다.
- **실제 이미지:** 학습에 없는 출처의 3,700장입니다. ImageNet·DOCCI 사진, CC3M 웹 이미지, WikiArt 회화, ImageNet-R 그림입니다.
- **오탐·탐지:** 학습 때 떼어 둔 실제 이미지로 정한 경계(`ai_score > 0.5`)에서 잰 값입니다.

| 조건 | AUROC | AI 탐지 | 실제 오탐 |
| --- | ---: | ---: | ---: |
| 원본 | 0.999 | 98.8% | 0.9% |
| JPEG 75로 다시 저장 | 0.998 | 94.9% | 0.4% |
| SNS식(0.75배 축소 + JPEG 80) | 0.998 | 93.4% | 0.5% |

비교 대상 두 가지입니다.
- 교사(PE-Core-G 고정 + 선형): 원본 98.7%, JPEG 75 91.0%, SNS식 95.1%
- 이전 서버: 원본 69.1%, JPEG 75 46.4%, SNS식 48.8%. 다른 OpenFake test shard에서 잰 값이라 직접 비교는 아닙니다.

### Codex(GPT-Image)로 직접 만든 이미지 (24차)

10장으로 시험했습니다.

| | 원본 | SNS식 |
| --- | ---: | ---: |
| 이 서버의 판정기 | 10/10 | 8/10 |
| 이전 서버 | 10/10 | 0/10 |

실제 사진 10장은 두 조건 모두 오탐이 없었습니다.

## 실행

```sh
UV_PYTHON_INSTALL_DIR=.python uv sync --extra dev
uv run uvicorn detector.app:app --app-dir src --port 8000
```

서버가 시작하려면 `models/`에 `student.onnx`(약 1.3GB)와 `student.json`(입력 크기, 경계값)이 있어야 합니다. 모델은 Git에 넣지 않으며, 아래 "재학습"으로 만듭니다. 이미지는 한 변이 256px 이상이어야 하고, 그보다 작으면 422를 반환합니다. 업로드는 최대 20MiB입니다.

## 한계

- **평가 데이터의 계열이 학습과 같습니다.** OpenFake·WikiArt·ImageNet-R·CC3M입니다. 실제로 유통된 이미지(Reddit 등)로는 아직 확인하지 않았습니다. Codex 시험은 10장뿐입니다.
- **출처별 오탐 차이가 있습니다.** 원본 기준으로 회화(WikiArt) 3.2%, 고해상도 정밀 사진(DOCCI) 2.0%입니다. 나머지 출처는 0.2–0.4%입니다.
- **약한 생성기:** Recraft-v3와 GPT-Image-2는 JPEG·SNS 처리 후 76–88%만 잡힙니다.
- **측정하지 않은 것:** 스크린샷, 신경망 처리를 거친 최신 폰 사진, 편집 사진의 오탐입니다.
- **라이선스:**
  - 학습 데이터 OpenFake는 CC BY-NC 4.0(비상업)입니다. Flickr30k도 비상업 연구용입니다. 그래서 상업 서비스에는 다른 데이터로 다시 학습해야 합니다.
  - PE-Core는 Apache 2.0입니다.

## 재학습

GPU 작업은 Kaggle(T4 2장)에서, 나머지는 노트북에서 합니다. Kaggle 명령줄 도구가 로그인돼 있어야 합니다(`kaggle auth login`).

```sh
UV_PYTHON_INSTALL_DIR=.python uv sync --all-extras
# 1. 원본 해상도 512px 조각 (OpenFake train 6 shard, validation·test 각 1 shard, COCO, Flickr30k)
uv run python scripts/prepare_data.py data/raw/core/*.parquet
uv run python scripts/prepare_data.py --real coco data/raw/val2017
uv run python scripts/prepare_data.py --real flickr30k data/raw/flickr30k/flickr30k-images
# 사람 그림과 옛 웹 이미지(WikiArt, ImageNet-R, CC3M)는 data/extra_crops/ 아래 extra_index.csv와 함께 둔다.
# 2. 조각을 tar로 묶어 이름을 *.bin으로 바꾼 뒤 Kaggle 비공개 데이터셋(synth-crops-bin)으로 올린다.
#    Kaggle은 *.tar는 스스로 풀어 수만 개 파일로 만드는데, 그러면 마운트가 실패하거나 느려진다.
# 3. 교사 특징: kaggle/siglip_features/main.py (COMPARE_PLAN = PE-Core-G, 학습 15,000 / test 6,000)
uv run python kaggle/push.py kaggle/siglip_features
# 4. 교사 점수: 5겹 교차 예측한 PE-Core-G 판정기 logit -> kaggle/distill/teacher.npz
kaggle kernels output chawanghyeon/siglip-features -p data/compare_peg
uv run python scripts/make_teacher.py data/compare_peg
# 5. 학생 학습과 채점 (약 1.5시간)
uv run python kaggle/push.py kaggle/distill
kaggle kernels output chawanghyeon/pe-distill -p data/distill
uv run python scripts/eval_distill.py data/distill data/compare_peg
# 6. ONNX 변환과 경계값 -> models/student.onnx, models/student.json
uv run python scripts/export_student.py
```

실험 과정과 기각한 방법은 [docs/experiments.md](docs/experiments.md)에 있습니다. 이전 프로젝트의 시도는 [docs/past_attempts.md](docs/past_attempts.md)에, 초기 조사 계획은 [docs/plan.md](docs/plan.md)에 있습니다.
