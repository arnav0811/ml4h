# Case Study: *dark*

**Classification:** drift candidate

## Frequency Profile

| Period | Frequency (per million) | Raw Count |
|--------|------------------------:|----------:|
| 1820-1860 | 343.6 | 796 |
| 1860-1900 | 366.5 | 507 |
| 1900-1940 | 467.8 | 413 |
| 1940-1980 | 463.1 | 406 |

## Semantic Drift Analysis

- **Overall centroid distance:** 0.0165
  *(95% CI: [0.0126, 0.0274])*
- **Average pairwise distance:** 0.3816
- **Neighbor Jaccard distance:** 0.235
- **Statistically significant drift:** Yes

### Drift Per Period Transition

| Transition | Centroid Distance | Avg Pairwise Distance |
|------------|------------------:|----------------------:|
| 1820-1860 -> 1860-1900 | 0.0059 | 0.3943 |
| 1860-1900 -> 1900-1940 | 0.0110 | 0.4056 |
| 1900-1940 -> 1940-1980 | 0.0041 | 0.3751 |

## Nearest Neighbors Per Period

These are the words used in similar contexts in each period. Shifts in this list directly evidence semantic drift.

**1820-1860:** `darkness` (0.90), `light` (0.81), `shadow` (0.78), `black` (0.78), `quiet` (0.76), `bright` (0.76), `pale` (0.75), `deep` (0.74), `mysterious` (0.73), `low` (0.73)

**1860-1900:** `darkness` (0.87), `black` (0.85), `light` (0.79), `quiet` (0.78), `pale` (0.77), `shadow` (0.77), `deep` (0.77), `mysterious` (0.76), `bright` (0.76), `low` (0.75)

**1900-1940:** `darkness` (0.90), `light` (0.80), `black` (0.80), `pale` (0.79), `quiet` (0.78), `shadow` (0.78), `soft` (0.75), `bright` (0.75), `deep` (0.74), `dull` (0.74)

**1940-1980:** `darkness` (0.88), `black` (0.84), `pale` (0.80), `bright` (0.79), `light` (0.79), `shadow` (0.79), `deep` (0.78), `soft` (0.78), `quiet` (0.77), `mysterious` (0.76)

## Example Sentences By Period

Representative usages from the corpus, with target word in **bold**.

### 1820-1860

- *rouncewell, says sir leicester with all the nature of a gentleman shining in him, it is late, and the roads are **dark**.*
- *he was not acquainted with arras; the streets were **dark**, and he walked on at random; but he seemed bent upon not asking the way of the passers-by.*
- *no golden light had ever been so precious as the gloom of this **dark** forest.*

### 1860-1900

- *you bet i ll follow him, if it s **dark**, huck.*
- *i discovered then, among other things, that these little people gathered into the great houses after **dark**, and slept in droves.*
- *lor , i wouldn t take less nor a quid a moment to stay there arter **dark**.*

### 1900-1940

- *calling encouraging words he shambles back with a furtive poacher s tread, dogged by the setter into a **dark** stalestunk corner.*
- *the eyes were very **dark** blue and steady.*
- *she sat holding it in her hand, while the music penetrated her whole being like an effulgence, warming and brightening the **dark** places of her soul.*

### 1940-1980

- *o, her mouth in the **dark**!*
- *soon, it was almost **dark**, but the look-out men still remained unset.*
- *and so saying the lighted tomahawk began flourishing about me in the **dark**.*
