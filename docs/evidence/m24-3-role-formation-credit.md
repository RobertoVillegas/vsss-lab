# M24.3 dynamic-role formation credit

Run 0013's final replay made the aggregate-policy defect visible:

| role | strike fraction | mean ball distance |
| --- | ---: | ---: |
| attacker | 0.875 | 0.194 m |
| support | 0.551 | 0.434 m |
| coverage | 0.482 | 0.455 m |

The roles affect position, but support and coverage still request a ball-contact primitive on
roughly half of their decisions. ADR 0023 keeps the terminal outcome shared and adds a dynamic,
identity-free support/coverage formation potential.

## Scale measured before training

Evaluated over all 18,000 role decisions in `iteration-002000.jsonl`, at coefficient `0.20`:

| quantity | value |
| --- | ---: |
| mean formation potential | 0.214 |
| potential p10 / p90 | 0.071 / 0.379 |
| mean absolute formation reward per decision | 0.000606 |
| absolute p95 | 0.001041 |
| carry reward in the same final run, absolute reported mean | 0.001646 |

Formation is therefore audible at about one third of carry without approaching the terminal
goal value. The coefficient's discounted-return bound is `0.20`, against carry `5` and goal
`10`.

## Throughput smoke benchmark

One CPU iteration, 16 environments and 4,096 frames, warm-started from run 0013 iteration 775:

| configuration | frames/s | matches/s |
| --- | ---: | ---: |
| formation off | 580.8 | 3.69 |
| formation at 0.20 | 610.5 | 3.88 |

This short paired smoke shows no measurable regression; the apparent 5.1 per cent improvement is
run noise, not a performance claim. A full run must select on match outcomes and role-resolved
behaviour, not throughput or total return.

## Acceptance signals for the next run

- coverage strike fraction falls materially below run 0013's 0.482 without collapsing to stop;
- support strike fraction falls below 0.551 while pass/receive and rotation do not regress;
- attacker remains the highest-strike responsibility;
- full-match goals per minute do not fall below 0.2 and draw rate does not exceed 0.70;
- the carry occupancy and goal rate that justified coefficient 5 are preserved.

