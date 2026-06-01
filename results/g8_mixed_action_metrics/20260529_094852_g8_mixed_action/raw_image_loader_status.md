# G8 Raw Image Loader Status

## Status: AVAILABLE (not run)

Raw images ARE accessible through HDF5 frame-reference resolution.
Frame references are strings of the form:
  `suite/file.hdf5:data/demo_N:obs/agentview_rgb:t`

A lazy raw-image loader (`LazyRawImageDataset`) has been implemented
in `src/eval/g8_mixed_action_metrics.py`.

Raw image CNN baselines are deferred to keep G8 scope focused on
split metric repair. They can be run in a future diagnostic.
