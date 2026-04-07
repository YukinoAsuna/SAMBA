# SDE

Official code for the KDD 2025 paper:

**SDE: A Simplified and Disentangled Dependency Encoding Framework for State Space Models in Time Series Forecasting**

Zixuan Weng, Jindong Han, Wenzhao Jiang, Hao Liu

Paper: https://arxiv.org/abs/2408.12068


## Usage

1. Install Pytorch and necessary dependencies.

```bash
pip install -r requirements.txt
```

2. The datasets can be obtained from the [iTransformer GitHub repository](https://github.com/thuml/iTransformer) or downloaded directly from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view).

3. Train and evaluate the model:

```bash
# Multivariate forecasting with SDE
bash ./scripts/SAMBA/weather.sh
```
