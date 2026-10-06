# AB-JChess PyTorch trainer

This directory contains the v11 feature encoder, model, native data loader,
distributed training helpers, checkpoint validation, and NNUE package exporter.

## Environment

Use Python 3.9 or newer and install the required packages:

```powershell
python -m pip install -r requirements.txt
```

Build the native loader after providing the JQv4 common reader through
`JQV4_COMMON_ROOT` when it is outside the source tree:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release `
  -DJQV4_COMMON_ROOT=<path-to-jieqi_v4_common>
cmake --build build --config Release
```

## Train

`train_v11.py` accepts JQv4 manifests or supported binary streams. Select one
of the v11 feature names and provide training and validation inputs:

```powershell
python .\train_v11.py <train-data> <validation-data> `
  --features HalfKAv2_hm_jieqi_v11^ --gpus 1
```

Keep checkpoints, logs, manifests, and training data outside this source
directory.

## Export

Export a v11 checkpoint with the matching feature name and probability tables:

```powershell
python .\serialize_v11.py <checkpoint> <output.nnue> `
  --features HalfKAv2_hm_jieqi_v11^ `
  --probability-score-to-mass <score-to-mass.i32le> `
  --probability-mass-to-score <mass-to-score.i32le>
```

The resulting package is accepted by the engine when supplied through UCI:

```text
setoption name EvalFile value C:\path\to\output.nnue
```
