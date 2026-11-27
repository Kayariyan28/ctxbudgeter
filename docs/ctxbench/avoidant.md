# CTXBench — agent `avoidant`, model `bench-model`

| mode | episodes | CIS | accuracy | retention | efficiency | recovery | critical | cum. input tokens | prefill (prefix cache) | prefill (suffix reuse) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| harness | 20 | 38.4 | 0.25 | 0.63 | 0.34 | n/a | 0 | 90,654 | 60,808 | 13,839 |
| unrestricted | 20 | 37.5 | 0.25 | 0.63 | 0.30 | n/a | 0 | 97,981 | 58,907 | 12,635 |
| guided | 20 | 37.5 | 0.25 | 0.63 | 0.30 | n/a | 0 | 97,981 | 58,907 | 12,635 |
| adaptive | 20 | 73.3 | 0.80 | 0.90 | 0.41 | 0.00 | 0 | 83,474 | 24,970 | 15,213 |

## By budget

| mode / budget | CIS | accuracy | retention |
|---|---:|---:|---:|
| adaptive/4000 | 66.4 | 0.70 | 0.87 |
| adaptive/8000 | 80.3 | 0.90 | 0.93 |
| guided/4000 | 26.4 | 0.10 | 0.53 |
| guided/8000 | 48.6 | 0.40 | 0.73 |
| harness/4000 | 26.4 | 0.10 | 0.53 |
| harness/8000 | 50.4 | 0.40 | 0.73 |
| unrestricted/4000 | 26.4 | 0.10 | 0.53 |
| unrestricted/8000 | 48.6 | 0.40 | 0.73 |

## Adaptive policy

- detected behaviours (calibration): avoidant, lossy
- selected candidate: tuned_without_lossy — status: validated
- rule: avoidant → governed fallback compaction at the hard limit; earlier pressure warning; 'compaction is expected' guidance

| candidate | calibration CIS | accuracy | critical | cum. input tokens |
|---|---:|---:|---:|---:|
| tuned_without_lossy | 69.6 | 0.70 | 0 | 81,833 |
| base_with_fallback | 69.6 | 0.70 | 0 | 81,833 |
| tuned | 52.5 | 0.40 | 0 | 86,396 |
| tuned_without_avoidant | 52.5 | 0.40 | 0 | 86,396 |
| base | 29.5 | 0.10 | 0 | 97,909 |

## Notes

- adaptive policy learned from 10 guided calibration session(s) on tasks disjoint from the evaluation set
- optimizer selected 'tuned_without_lossy' from 5 candidate policies (calibration CIS 69.6 vs guided 29.5); status: validated
