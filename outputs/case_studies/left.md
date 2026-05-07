# Case Study: *left*

**Classification:** drift candidate

## Frequency Profile

| Period | Frequency (per million) | Raw Count |
|--------|------------------------:|----------:|
| 1820-1860 | 713.2 | 1652 |
| 1860-1900 | 897.8 | 1242 |
| 1900-1940 | 711.4 | 628 |
| 1940-1980 | 630.7 | 553 |

## Semantic Drift Analysis

- **Overall centroid distance:** 0.0160
  *(95% CI: [0.0131, 0.0361])*
- **Average pairwise distance:** 0.5729
- **Neighbor Jaccard distance:** 0.571
- **Statistically significant drift:** No

### Drift Per Period Transition

| Transition | Centroid Distance | Avg Pairwise Distance |
|------------|------------------:|----------------------:|
| 1820-1860 -> 1860-1900 | 0.0078 | 0.5394 |
| 1860-1900 -> 1900-1940 | 0.0227 | 0.5494 |
| 1900-1940 -> 1940-1980 | 0.0074 | 0.5803 |

## Nearest Neighbors Per Period

These are the words used in similar contexts in each period. Shifts in this list directly evidence semantic drift.

**1820-1860:** `leave` (0.88), `leaving` (0.87), `gone` (0.83), `lost` (0.82), `dropped` (0.80), `passed` (0.80), `taken` (0.80), `made` (0.80), `remained` (0.79), `arrived` (0.79)

**1860-1900:** `leaving` (0.88), `leave` (0.85), `gone` (0.80), `passed` (0.79), `moved` (0.78), `remained` (0.78), `lost` (0.77), `taken` (0.77), `reached` (0.77), `arrived` (0.77)

**1900-1940:** `leaving` (0.88), `leave` (0.86), `remained` (0.82), `gone` (0.82), `passed` (0.79), `returned` (0.79), `taken` (0.79), `lost` (0.78), `followed` (0.78), `made` (0.78)

**1940-1980:** `leave` (0.85), `leaving` (0.85), `gone` (0.80), `lost` (0.79), `right` (0.78), `remained` (0.78), `passed` (0.77), `moved` (0.77), `returned` (0.77), `taken` (0.77)

## Example Sentences By Period

Representative usages from the corpus, with target word in **bold**.

### 1820-1860

- *my health, severely impaired when i **left** england, was quite restored.*
- *nevertheless, the altar has been **left** there an altar of unpolished wood, placed against a background of roughhewn stone.*
- *i, however, had dispensed with the two middle names long before i **left** maryland so that i was generally known by the name of frederick bailey.*

### 1860-1900

- *presently, joe came back, saying that the man was gone, but that he, joe, had **left** word at the three jolly bargemen concerning the notes.*
- *as soon as he had **left**, he rushed to the screen and drew it back.*
- *nevertheless i **left** that gallery greatly elated.*

### 1900-1940

- *in the ditch beside the road, right side up, but violently shorn of one wheel, rested a new coup which had **left** gatsby s drive not two minutes before.*
- *they had **left** the big road and turned into a level plain which had formerly been an old meadow.*
- *she would not consent to remain with edna, for monsieur ratignolle was alone, and he detested above all things to be **left** alone.*

### 1940-1980

- *but it is only found on the sinister side, which has an ill effect, giving its owner something analogous to the aspect of a clumsy **left**-handed man.*
- *darcy, she **left** elizabeth to walk by herself.*
- *the cane moved out trembling to the **left**.*
