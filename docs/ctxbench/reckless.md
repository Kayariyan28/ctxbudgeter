# CTXBench — agent `reckless`, model `bench-model`

| mode | episodes | CIS | accuracy | retention | efficiency | recovery | critical | cum. input tokens | prefill (prefix cache) | prefill (suffix reuse) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| harness | 20 | 38.4 | 0.25 | 0.63 | 0.34 | n/a | 0 | 90,654 | 60,808 | 13,839 |
| unrestricted | 20 | 0.0 | 1.00 | 1.00 | 0.62 | 1.00 | 20 | 52,579 | 16,955 | 16,955 |
| guided | 20 | 0.0 | 1.00 | 1.00 | 0.62 | 1.00 | 20 | 52,579 | 16,955 | 16,955 |
| adaptive | 20 | 76.4 | 0.85 | 0.90 | 0.39 | 1.00 | 0 | 85,888 | 27,384 | 17,628 |

## By budget

| mode / budget | CIS | accuracy | retention |
|---|---:|---:|---:|
| adaptive/4000 | 72.8 | 0.80 | 0.87 |
| adaptive/8000 | 80.0 | 0.90 | 0.93 |
| guided/4000 | 0.0 | 1.00 | 1.00 |
| guided/8000 | 0.0 | 1.00 | 1.00 |
| harness/4000 | 26.4 | 0.10 | 0.53 |
| harness/8000 | 50.4 | 0.40 | 0.73 |
| unrestricted/4000 | 0.0 | 1.00 | 1.00 |
| unrestricted/8000 | 0.0 | 1.00 | 1.00 |

## Adaptive policy

- detected behaviours (calibration): over_compressing, unsafe
- selected candidate: base_with_fallback — status: validated
- rule: base policy + governed fallback compaction

| candidate | calibration CIS | accuracy | critical | cum. input tokens |
|---|---:|---:|---:|---:|
| base_with_fallback | 74.6 | 0.80 | 0 | 84,485 |
| tuned_without_over_compressing | 74.2 | 0.80 | 0 | 87,884 |
| tuned_without_unsafe | 68.7 | 0.70 | 0 | 88,713 |
| tuned | 68.1 | 0.70 | 0 | 91,795 |
| base | 29.5 | 0.10 | 0 | 97,909 |

## Notes

- adaptive policy learned from 10 guided calibration session(s) on tasks disjoint from the evaluation set
- optimizer selected 'base_with_fallback' from 5 candidate policies (calibration CIS 74.6 vs guided 0.0); status: validated
