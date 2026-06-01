# Representation-Action Retrieval Report

G6 diagnostic: whether each representation encodes control-relevant state vs. demo phase.

## Metrics

| Label | Var Mean | Adj Cos Sim | NN Timestep Acc | NN Action MSE | Lat-Act Corr | PCA Top-5 |
|---|---:|---:|---:|---:|---:|---:|
| dino_cls | 7.165527e-02 | 0.999359 | 0.8544 | 9.166629e-02 | 0.0986 | 0.9502 |
| proprio_only_state | 2.841922e-03 | 0.999979 | 1.0000 | 1.758555e-02 | 0.3772 | 0.9974 |

## Interpretation Guide

- **High NN timestep accuracy** (close to 1.0): representation mainly encodes demo phase/time,
  not fine-grained control state. This would mean the representation collapses to a phase indicator.
- **Low NN action MSE**: representation captures action-relevant information well.
- **High latent-action distance correlation**: latent distance tracks action distance,
  suggesting the representation encodes control-relevant variation.
- **High PCA concentration**: most variance in few dimensions, possibly indicating
  low effective dimensionality.

These metrics diagnose representation quality for control, not policy quality.
