# Case Study: *son*

**Classification:** drift candidate

## Frequency Profile

| Period | Frequency (per million) | Raw Count |
|--------|------------------------:|----------:|
| 1820-1860 | 266.8 | 618 |
| 1860-1900 | 393.2 | 544 |
| 1900-1940 | 564.1 | 498 |
| 1940-1980 | 563.4 | 494 |

## Semantic Drift Analysis

- **Overall centroid distance:** 0.0419
  *(95% CI: [0.0339, 0.0562])*
- **Average pairwise distance:** 0.3780
- **Neighbor Jaccard distance:** 0.333
- **Statistically significant drift:** Yes

### Drift Per Period Transition

| Transition | Centroid Distance | Avg Pairwise Distance |
|------------|------------------:|----------------------:|
| 1820-1860 -> 1860-1900 | 0.0065 | 0.3324 |
| 1860-1900 -> 1900-1940 | 0.0484 | 0.3844 |
| 1900-1940 -> 1940-1980 | 0.0018 | 0.3797 |

## Nearest Neighbors Per Period

These are the words used in similar contexts in each period. Shifts in this list directly evidence semantic drift.

**1820-1860:** `daughter` (0.83), `boy` (0.82), `brother` (0.80), `child` (0.80), `children` (0.79), `father` (0.78), `boys` (0.75), `man` (0.74), `youth` (0.73), `born` (0.72)

**1860-1900:** `child` (0.83), `daughter` (0.83), `boy` (0.83), `children` (0.80), `father` (0.78), `brother` (0.78), `boys` (0.76), `husband` (0.75), `man` (0.72), `youth` (0.72)

**1900-1940:** `daughter` (0.82), `boy` (0.80), `brother` (0.79), `child` (0.78), `father` (0.77), `children` (0.76), `youth` (0.73), `boys` (0.71), `man` (0.71), `born` (0.71)

**1940-1980:** `daughter` (0.82), `boy` (0.81), `child` (0.78), `brother` (0.76), `father` (0.75), `children` (0.75), `youth` (0.73), `man` (0.72), `boys` (0.71), `born` (0.70)

## Example Sentences By Period

Representative usages from the corpus, with target word in **bold**.

### 1820-1860

- *if i ever get to heaven it won t be for being a good **son** to a widowed mother; i say no more.*
- *father and **son** entered the labyrinth of walks which leads to the grand flight of steps near the clump of trees on the side of the rue madame.*
- *and the same with a king s **son**; it don t make no difference whether he s a natural one or an unnatural one.*

### 1860-1900

- *this assurance delighted morrel, who took leave of villefort, and hastened to announce to old dant s that he would soon see his **son**.*
- *his **son** was bred in the service of his country, and agatha had ranked with ladies of the highest distinction.*
- *but that does not affect the **son**.*

### 1900-1940

- *he was grieved at what was done, and showed to the husband, the **son** of juno, 28 the wrong done to his bed, and the place of the intrigue.*
- *her **son**-in-law brought them every year to the lakes and they used to go fishing.*
- *his **son** jules is with him jules, who wants to marry her.*

### 1940-1980

- *what now, **son** of hyperion, 31 does thy beauty, thy heat, and thy radiant light avail thee?*
- *her voice had a catch in it like her **son** s and she stuttered slightly.*
- *in my opinion, the younger **son** of an earl can know very little of either.*
