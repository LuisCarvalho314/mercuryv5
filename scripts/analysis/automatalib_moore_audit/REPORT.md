# AutomataLib oracle Moore-machine certification

Each machine uses physical position as state, north/east/south/west as input, and the cached range-1 cardinal-distance observation vector as state output. The selected walks are 0–2 from the canonical-v4 cache in `ambiguity_response_ablation`.

| Level | Physical states | Moore classes | State–action rows | Singleton classes | Aliased pairs separated | Words in W | Max word length |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 13 | 8 | 8 | 32 | yes | 2/2 | 3 | 1 |
| 20 | 5 | 5 | 20 | yes | 3/3 | 3 | 1 |
| 21 | 4 | 4 | 16 | yes | 1/1 | 2 | 1 |
| 22 | 12 | 12 | 48 | yes | 3/3 | 2 | 1 |

## Classes and witnesses

Witness action IDs: 0=north, 1=east, 2=south, 3=west.

- **Level 13:** classes {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}.
  Aliased-pair witnesses: (1, 6) → [1]; (3, 4) → [0].
- **Level 20:** classes {0}, {1}, {2}, {3}, {4}.
  Aliased-pair witnesses: (1, 2) → [3]; (1, 3) → [1]; (2, 3) → [1].
- **Level 21:** classes {0}, {1}, {2}, {3}.
  Aliased-pair witnesses: (1, 2) → [1].
- **Level 22:** classes {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}.
  Aliased-pair witnesses: (1, 2) → [1]; (5, 6) → [1]; (9, 10) → [1].

## Global characterizing sets

The empty word ε compares the current state outputs; a Moore trace includes that output before any action.

- **Level 13:** W0=ε (length 0); W1=[0] (length 1); W2=[1] (length 1).
  Independent replay separates 28/28 physical-state pairs; mean/median word length 0.666667/1.
  Observation-aliased pairs distinguished by: (1, 6): W2; (3, 4): W1.
- **Level 20:** W0=ε (length 0); W1=[3] (length 1); W2=[1] (length 1).
  Independent replay separates 10/10 physical-state pairs; mean/median word length 0.666667/1.
  Observation-aliased pairs distinguished by: (1, 2): W1; (1, 3): W1, W2; (2, 3): W2.
- **Level 21:** W0=ε (length 0); W1=[1] (length 1).
  Independent replay separates 6/6 physical-state pairs; mean/median word length 0.5/0.5.
  Observation-aliased pairs distinguished by: (1, 2): W1.
- **Level 22:** W0=ε (length 0); W1=[1] (length 1).
  Independent replay separates 66/66 physical-state pairs; mean/median word length 0.5/0.5.
  Observation-aliased pairs distinguished by: (1, 2): W1; (5, 6): W1; (9, 10): W1.

## What this establishes

All physical states and all four actions per state occur in the selected walks. Their outputs and successors are deterministic and agree with the maze geometry. AutomataLib 13.0.0 minimizes the resulting complete Moore machines; its state-pair separating words give the listed witnesses. The Java runner checks that pairwise classes match its minimized count. AutomataLib constructs a global characterizing set W; Python independently replays every word and verifies all physical-state pairs. It also replays every pairwise witness. Cache hashes, full tables, classes, words, and pair coverage are in `RESULTS.json`.

## What this does not establish

This is an oracle statement about these fixed benchmark machines and their privileged physical states. It establishes behavioral irreducibility and finite distinguishing action sequences. It does not reconstruct an epsilon-transducer, equate physical states with its causal states, show that a learner can infer these states from sensorimotor history, or provide a baseline comparison with MERCURY, CSCG, TMGWR, or transCSSR. Completeness is conditional on the benchmark layout, cached ground truth, and checked alignment.

AutomataLib APIs: [CompactMoore](https://learnlib.de/automatalib/maven-site/13.0.0/apidocs/net.automatalib.core/net/automatalib/automaton/transducer/impl/CompactMoore.html), [HopcroftMinimizer](https://learnlib.de/automatalib/maven-site/13.0.0/apidocs/net.automatalib.util/net/automatalib/util/automaton/minimizer/HopcroftMinimizer.html), [Automata.characterizingSet](https://learnlib.de/automatalib/maven-site/13.0.0/apidocs/net.automatalib.util/net/automatalib/util/automaton/Automata.html), [CharacterizingSets](https://learnlib.de/automatalib/maven-site/13.0.0/apidocs/net.automatalib.util/net/automatalib/util/automaton/equivalence/CharacterizingSets.html), and [Automata.findSeparatingWord](https://learnlib.de/automatalib/maven-site/13.0.0/apidocs/net.automatalib.util/net/automatalib/util/automaton/Automata.html).
