# Case Study: *hands*

**Classification:** drift candidate

## Frequency Profile

| Period | Frequency (per million) | Raw Count |
|--------|------------------------:|----------:|
| 1820-1860 | 666.5 | 1544 |
| 1860-1900 | 807.5 | 1117 |
| 1900-1940 | 649.1 | 573 |
| 1940-1980 | 675.2 | 592 |

## Semantic Drift Analysis

- **Overall centroid distance:** 0.0167
  *(95% CI: [0.0140, 0.0288])*
- **Average pairwise distance:** 0.4617
- **Neighbor Jaccard distance:** 0.333
- **Statistically significant drift:** Yes

### Drift Per Period Transition

| Transition | Centroid Distance | Avg Pairwise Distance |
|------------|------------------:|----------------------:|
| 1820-1860 -> 1860-1900 | 0.0055 | 0.4649 |
| 1860-1900 -> 1900-1940 | 0.0092 | 0.4555 |
| 1900-1940 -> 1940-1980 | 0.0063 | 0.4343 |

## Nearest Neighbors Per Period

These are the words used in similar contexts in each period. Shifts in this list directly evidence semantic drift.

**1820-1860:** `hand` (0.84), `fingers` (0.83), `arms` (0.80), `feet` (0.74), `touch` (0.70), `hold` (0.69), `arm` (0.69), `legs` (0.69), `holding` (0.69), `eyes` (0.69)

**1860-1900:** `fingers` (0.87), `hand` (0.85), `arms` (0.78), `touch` (0.72), `touched` (0.71), `arm` (0.71), `legs` (0.70), `holding` (0.70), `lips` (0.70), `shaking` (0.69)

**1900-1940:** `hand` (0.86), `fingers` (0.85), `arms` (0.81), `feet` (0.75), `arm` (0.73), `touch` (0.71), `holding` (0.71), `touched` (0.70), `laid` (0.69), `eyes` (0.69)

**1940-1980:** `hand` (0.85), `fingers` (0.85), `arms` (0.82), `arm` (0.77), `feet` (0.75), `hold` (0.73), `legs` (0.73), `held` (0.72), `lips` (0.72), `heads` (0.71)

## Example Sentences By Period

Representative usages from the corpus, with target word in **bold**.

### 1820-1860

- *exclaimed joseph, lifting his **hands** and eyes in horror.*
- *who are almost young maidens, thin, feverish, with sunburnt **hands**, covered with freckles, crowned with poppies and ears of rye, gay, haggard, barefooted.*
- *his **hands** were hardened by toil, and not his alone, but those also of mrs.*

### 1860-1900

- *shame on the coward, caitiff **hands** that smote their lord or with a kiss betrayed him to the rabble-rout of fawning priests no friends of his.*
- *her little **hands** stretched blindly out, and appeared to be seeking for him.*
- *we passed each other flowers, and she kissed my **hands**.*

### 1900-1940

- *i know they are in better **hands** than mine.*
- *a black woman, wiping her **hands** upon her apron, was close at his heels.*
- *buck mulligan, walking forward again, raised his **hands**.*

### 1940-1980

- *there was nothing he liked, he said, so much as looking at a nice young girl, at her nice white **hands** and her beautiful soft hair.*
- *hear him, hear him now, cried peleg, marching across the cabin, and thrusting his **hands** far down into his pockets, hear him, all of ye.*
- *he bent down to her, his **hands** on his knees.*
