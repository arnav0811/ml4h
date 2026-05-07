# Case Study: *long*

**Classification:** drift candidate

## Frequency Profile

| Period | Frequency (per million) | Raw Count |
|--------|------------------------:|----------:|
| 1820-1860 | 1219.1 | 2824 |
| 1860-1900 | 1193.5 | 1651 |
| 1900-1940 | 1231.3 | 1087 |
| 1940-1980 | 1309.4 | 1148 |

## Semantic Drift Analysis

- **Overall centroid distance:** 0.0157
  *(95% CI: [0.0131, 0.0303])*
- **Average pairwise distance:** 0.5116
- **Neighbor Jaccard distance:** 0.500
- **Statistically significant drift:** No

### Drift Per Period Transition

| Transition | Centroid Distance | Avg Pairwise Distance |
|------------|------------------:|----------------------:|
| 1820-1860 -> 1860-1900 | 0.0085 | 0.5119 |
| 1860-1900 -> 1900-1940 | 0.0185 | 0.5132 |
| 1900-1940 -> 1940-1980 | 0.0154 | 0.5155 |

## Nearest Neighbors Per Period

These are the words used in similar contexts in each period. Shifts in this list directly evidence semantic drift.

**1820-1860:** `years` (0.76), `deep` (0.75), `far` (0.75), `great` (0.74), `length` (0.74), `hard` (0.73), `short` (0.73), `last` (0.72), `vast` (0.72), `large` (0.71)

**1860-1900:** `deep` (0.76), `far` (0.76), `short` (0.76), `broad` (0.75), `large` (0.75), `tall` (0.75), `heavy` (0.74), `wide` (0.74), `thick` (0.73), `vast` (0.73)

**1900-1940:** `length` (0.79), `short` (0.75), `deep` (0.74), `wide` (0.74), `years` (0.74), `strong` (0.73), `hard` (0.72), `minute` (0.72), `large` (0.72), `broad` (0.72)

**1940-1980:** `short` (0.80), `deep` (0.77), `length` (0.77), `minute` (0.76), `broad` (0.75), `thick` (0.75), `heavy` (0.75), `wide` (0.75), `tall` (0.75), `strong` (0.75)

## Example Sentences By Period

Representative usages from the corpus, with target word in **bold**.

### 1820-1860

- *when lady catherine and her daughter had played as **long** as they chose, the tables were broken up, the carriage was offered to mrs.*
- *shouts some street arab, how **long** has it been customary for doctors to carry home their own work?*
- *they have kept thy better part in bondage too **long** already!*

### 1860-1900

- *and his brother nikolay s gentleness did in fact not last out for **long**.*
- *a **long**, lingering, colossal sigh followed, and his heart broke.*
- *he had ticked so **long** that he now went on ticking without knowing that he was doing it.*

### 1900-1940

- *and so now you have come into the town, and have taken this **long** journey in winter that was plucky of you.*
- *the mandolin players had **long** since stolen away.*
- *he s out of that **long** ago, nosey flynn said.*

### 1940-1980

- *though no coward, he has never yet shown any part of him but his back, which rises in a **long** sharp ridge.*
- *but do you always write such charming **long** letters to her, mr.*
- *his confession would be **long**, **long**.*
