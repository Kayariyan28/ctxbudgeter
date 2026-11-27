# CTXBench — agent `anxious`, model `bench-model`

| mode | episodes | CIS | accuracy | retention | efficiency | recovery | critical | cum. input tokens | prefill (prefix cache) | prefill (suffix reuse) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| harness | 20 | 38.4 | 0.25 | 0.63 | 0.34 | n/a | 0 | 90,654 | 60,808 | 13,839 |
| unrestricted | 20 | 82.6 | 0.85 | 0.90 | 0.63 | 1.00 | 0 | 49,917 | 32,255 | 17,542 |
| guided | 20 | 80.3 | 0.85 | 0.90 | 0.53 | 1.00 | 0 | 63,602 | 28,282 | 17,443 |
| adaptive | 20 | 80.0 | 0.85 | 0.90 | 0.53 | 1.00 | 0 | 64,320 | 28,233 | 17,255 |

## By budget

| mode / budget | CIS | accuracy | retention |
|---|---:|---:|---:|
| adaptive/4000 | 75.7 | 0.80 | 0.87 |
| adaptive/8000 | 84.3 | 0.90 | 0.93 |
| guided/4000 | 76.0 | 0.80 | 0.87 |
| guided/8000 | 84.6 | 0.90 | 0.93 |
| harness/4000 | 26.4 | 0.10 | 0.53 |
| harness/8000 | 50.4 | 0.40 | 0.73 |
| unrestricted/4000 | 78.8 | 0.80 | 0.87 |
| unrestricted/8000 | 86.5 | 0.90 | 0.93 |

## Adaptive policy

- detected behaviours (calibration): over_compressing
- selected candidate: tuned_without_over_compressing — status: validated
- rule: no failure behaviours detected → base policy retained

| candidate | calibration CIS | accuracy | critical | cum. input tokens |
|---|---:|---:|---:|---:|
| tuned_without_over_compressing | 78.4 | 0.80 | 0 | 64,994 |
| base_with_fallback | 78.4 | 0.80 | 0 | 64,994 |
| tuned | 68.6 | 0.70 | 0 | 89,075 |

## Notes

- adaptive policy learned from 10 guided calibration session(s) on tasks disjoint from the evaluation set
- optimizer selected 'tuned_without_over_compressing' from 3 candidate policies (calibration CIS 78.4 vs guided 78.7); status: validated
