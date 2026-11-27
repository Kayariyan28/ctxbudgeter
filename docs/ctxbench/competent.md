# CTXBench — agent `competent`, model `bench-model`

| mode | episodes | CIS | accuracy | retention | efficiency | recovery | critical | cum. input tokens | prefill (prefix cache) | prefill (suffix reuse) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| harness | 20 | 38.4 | 0.25 | 0.63 | 0.34 | n/a | 0 | 90,654 | 60,808 | 13,839 |
| unrestricted | 20 | 77.9 | 0.85 | 0.90 | 0.43 | 1.00 | 0 | 79,386 | 28,131 | 18,119 |
| guided | 20 | 77.8 | 0.85 | 0.90 | 0.43 | 1.00 | 0 | 80,059 | 28,236 | 18,235 |
| adaptive | 20 | 77.8 | 0.85 | 0.90 | 0.43 | 1.00 | 0 | 80,122 | 28,174 | 18,173 |

## By budget

| mode / budget | CIS | accuracy | retention |
|---|---:|---:|---:|
| adaptive/4000 | 74.6 | 0.80 | 0.87 |
| adaptive/8000 | 81.0 | 0.90 | 0.93 |
| guided/4000 | 74.6 | 0.80 | 0.87 |
| guided/8000 | 81.0 | 0.90 | 0.93 |
| harness/4000 | 26.4 | 0.10 | 0.53 |
| harness/8000 | 50.4 | 0.40 | 0.73 |
| unrestricted/4000 | 74.8 | 0.80 | 0.87 |
| unrestricted/8000 | 81.1 | 0.90 | 0.93 |

## Adaptive policy

- detected behaviours (calibration): none
- selected candidate: tuned — status: validated
- rule: no failure behaviours detected → base policy retained

| candidate | calibration CIS | accuracy | critical | cum. input tokens |
|---|---:|---:|---:|---:|
| tuned | 75.5 | 0.80 | 0 | 80,881 |
| base_with_fallback | 75.5 | 0.80 | 0 | 80,881 |

## Notes

- adaptive policy learned from 10 guided calibration session(s) on tasks disjoint from the evaluation set
- optimizer selected 'tuned' from 2 candidate policies (calibration CIS 75.5 vs guided 75.5); status: validated
