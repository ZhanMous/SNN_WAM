# G9 Gripper Transition Dataset Audit

## Demos Analyzed: 50
## Total Transitions: 110
## Average Transitions Per Demo: 2.20

## Per-Demo Statistics (first 10)

| Demo | Length | Transitions | Open% | Close% |
|---|---:|---:|---:|---:|
| demo_0 | 103 | 2 | 62.1% | 37.9% |
| demo_1 | 123 | 2 | 55.3% | 44.7% |
| demo_10 | 109 | 2 | 61.5% | 38.5% |
| demo_11 | 101 | 1 | 68.3% | 31.7% |
| demo_12 | 124 | 2 | 55.6% | 44.4% |
| demo_13 | 122 | 4 | 61.5% | 38.5% |
| demo_14 | 114 | 2 | 55.3% | 44.7% |
| demo_15 | 136 | 2 | 42.6% | 57.4% |
| demo_16 | 105 | 2 | 55.2% | 44.8% |
| demo_17 | 164 | 4 | 50.0% | 50.0% |

## Assessment

- With 2.2 transitions per demo, gripper transition F1 is uninformative for single-demo diagnostics.
- Low transition count means F1 is dominated by the 'stay' class.
- Multi-demo evaluation would provide more meaningful gripper transition statistics.
