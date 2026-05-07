# Case Study: *right*

**Classification:** drift candidate

## Frequency Profile

| Period | Frequency (per million) | Raw Count |
|--------|------------------------:|----------:|
| 1820-1860 | 888.4 | 2058 |
| 1860-1900 | 804.6 | 1113 |
| 1900-1940 | 802.0 | 708 |
| 1940-1980 | 837.2 | 734 |

## Semantic Drift Analysis

- **Overall centroid distance:** 0.0157
  *(95% CI: [0.0151, 0.0316])*
- **Average pairwise distance:** 0.5752
- **Neighbor Jaccard distance:** 0.696
- **Statistically significant drift:** No

### Drift Per Period Transition

| Transition | Centroid Distance | Avg Pairwise Distance |
|------------|------------------:|----------------------:|
| 1820-1860 -> 1860-1900 | 0.0123 | 0.5862 |
| 1860-1900 -> 1900-1940 | 0.0109 | 0.5893 |
| 1900-1940 -> 1940-1980 | 0.0168 | 0.5682 |

## Nearest Neighbors Per Period

These are the words used in similar contexts in each period. Shifts in this list directly evidence semantic drift.

**1820-1860:** `straight` (0.80), `wrong` (0.77), `perfect` (0.76), `natural` (0.74), `proper` (0.74), `put` (0.73), `bent` (0.73), `fine` (0.73), `directly` (0.73), `went` (0.72)

**1860-1900:** `wrong` (0.82), `proper` (0.72), `point` (0.70), `left` (0.70), `true` (0.69), `directly` (0.69), `position` (0.68), `serious` (0.68), `believed` (0.68), `bent` (0.68)

**1900-1940:** `wrong` (0.77), `fine` (0.70), `straight` (0.70), `left` (0.70), `fair` (0.68), `best` (0.68), `proper` (0.68), `perfectly` (0.67), `bent` (0.67), `directly` (0.66)

**1940-1980:** `left` (0.78), `wrong` (0.73), `straight` (0.73), `forward` (0.70), `proper` (0.70), `front` (0.70), `natural` (0.70), `bent` (0.70), `directly` (0.69), `sense` (0.69)

## Example Sentences By Period

Representative usages from the corpus, with target word in **bold**.

### 1820-1860

- *in fact, i have not the **right** to give her to you.*
- *it s natural and **right** for em to talk different from each other, ain t it?*
- *when i arrived there, i walked **right** into the arms of a police-officer who was coming out, and only managed to disarm his suspicions by pretending to be hopelessly drunk.*

### 1860-1900

- *she ll be there all **right**.*
- *it has its divine **right** of sovereignty.*
- *be all **right** in a minute.*

### 1900-1940

- *but she rushes onward, about to confound both **right** and wrong, and is wholly occupied in the contrivance of revenge.*
- *of course, they say it was all **right**, that it contained nothing, i mean.*
- *i ll speak to bunbury, aunt augusta, if he is still conscious, and i think i can promise you he ll be all **right** by saturday.*

### 1940-1980

- *a limpid fountain ran murmuring on the **right** hand with its little stream, having its spreading channels edged with a border of grass.*
- *look ye; when captain ahab is all **right**, then this left arm of mine will be all **right**; not before.*
- *besides, i have a perfect **right** to be christened if i like.*
