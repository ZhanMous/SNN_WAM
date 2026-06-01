# G7 Raw Image Frame-Reference Audit

Sample reference: `libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5:data/demo_0:obs/agentview_rgb:0`
Type: `str`
Is string reference: True
Can dereference: True
Dereference method: hdf5_dataset_path

## Assessment

- Frame references are HDF5 dataset paths of the form:
  `suite/file.hdf5:data/demo_N:obs/agentview_rgb:t`
- These CAN be resolved to raw 128x128x3 RGB uint8 arrays by reading the HDF5 file.
- The current pipeline does NOT resolve them because it uses pre-extracted DINO latents.
- A lazy raw-image loader could be implemented to resolve these on demand.
- The raw-image CNN baseline in G6 was skipped because the dataset returns frame references
  (strings), not raw pixel arrays.

## Resolution Path

1. Parse the frame reference string
2. Open the HDF5 file at the referenced path
3. Read `data/demo_N/obs/agentview_rgb[t]`
4. Return the uint8 array

## Conclusion

Raw images ARE accessible through frame-reference resolution.
A lazy loader can be implemented to enable raw-image CNN baselines.
