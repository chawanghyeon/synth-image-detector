"""AI-generated image detector: a fine-tuned PE-Core-L student served through ONNX.

Subpackages, by what they may import:
  preprocess, model_card   numpy + PIL only; shared by every stage
  serving                  + onnxruntime, fastapi (the deployed server, no torch)
  data                     + huggingface_hub, pyarrow (builds the crop datasets on Kaggle CPU)
  training                 + torch, timm (Kaggle GPU jobs, ONNX export, laptop scoring)
  evaluation               numpy only (reads score files written by training jobs)
  kaggle_jobs, cli         local tooling: push/fetch Kaggle jobs, the `detector` command
"""
