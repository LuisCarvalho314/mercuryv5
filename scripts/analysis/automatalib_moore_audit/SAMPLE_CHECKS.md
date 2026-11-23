# Sample-level Moore equivalence and coverage

Each 2,000-step parquet is replayed from its privileged metadata start position. Each 10,000-step evaluation walk is checked against its cached physical states. Observations, actions, transitions, and collisions must match the certified Moore machine.

| Level | 2,000-step datasets consistent | Datasets certified | 10,000-step walks consistent | Walks certified | Moore classes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 13 | 91/91 | 56/91 | 100/100 | 100/100 | 8 |
| 20 | 91/91 | 61/91 | 100/100 | 100/100 | 5 |
| 21 | 91/91 | 61/91 | 100/100 | 100/100 | 4 |
| 22 | 60/60 | 57/60 | 100/100 | 100/100 | 12 |

The 235 complete datasets and 400 complete evaluation walks each supply a full sampled Moore table. The tables were deduplicated into 4 distinct machines, then minimized with AutomataLib. Every resulting class is a singleton. Per-sample classes and separating words are in `SAMPLE_CHECKS.json`.

A sample with missing state–action pairs cannot by itself establish the minimized machine. Its observed rows still agree with the complete oracle machine; its equivalence fields are null. Full per-sample coverage and failure details are in `SAMPLE_CHECKS.json`.

The 2,000-step parquet files do not contain physical-state paths. Their paths are reconstructed from privileged initial positions and the benchmark maze layout; this is an oracle check, not a learner evaluation.
