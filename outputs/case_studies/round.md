# Case Study: *round*

**Classification:** drift candidate

## Frequency Profile

| Period | Frequency (per million) | Raw Count |
|--------|------------------------:|----------:|
| 1820-1860 | 686.8 | 1591 |
| 1860-1900 | 761.2 | 1053 |
| 1900-1940 | 685.3 | 605 |
| 1940-1980 | 883.9 | 775 |

## Semantic Drift Analysis

- **Overall centroid distance:** 0.0164
  *(95% CI: [0.0138, 0.0295])*
- **Average pairwise distance:** 0.4887
- **Neighbor Jaccard distance:** 0.636
- **Statistically significant drift:** No

### Drift Per Period Transition

| Transition | Centroid Distance | Avg Pairwise Distance |
|------------|------------------:|----------------------:|
| 1820-1860 -> 1860-1900 | 0.0092 | 0.4787 |
| 1860-1900 -> 1900-1940 | 0.0193 | 0.5032 |
| 1900-1940 -> 1940-1980 | 0.0074 | 0.4816 |

## Nearest Neighbors Per Period

These are the words used in similar contexts in each period. Shifts in this list directly evidence semantic drift.

**1820-1860:** `towards` (0.75), `large` (0.72), `sort` (0.71), `upon` (0.71), `seized` (0.71), `whole` (0.71), `altogether` (0.70), `fixed` (0.70), `great` (0.69), `formed` (0.69)

**1860-1900:** `towards` (0.72), `seized` (0.68), `sort` (0.67), `afterwards` (0.67), `little` (0.67), `upon` (0.67), `forward` (0.67), `fixed` (0.66), `along` (0.66), `altogether` (0.66)

**1900-1940:** `towards` (0.73), `along` (0.72), `upon` (0.72), `whole` (0.70), `fixed` (0.70), `making` (0.70), `raised` (0.70), `covered` (0.70), `among` (0.69), `set` (0.69)

**1940-1980:** `towards` (0.72), `upon` (0.71), `along` (0.70), `whole` (0.70), `altogether` (0.68), `close` (0.68), `struck` (0.68), `forward` (0.67), `seized` (0.67), `cast` (0.67)

## Example Sentences By Period

Representative usages from the corpus, with target word in **bold**.

### 1820-1860

- *after two or three such adjurations, jo lifts up his head again, looks **round** the court again, and says in a low voice, well, i ll tell you something.*
- *he added, turning quickly **round** to the four medical men who were assembled.*
- *this knob, which was **round** and of polished brass, shone like a terrible star for him.*

### 1860-1900

- *a bee flew in and buzzed **round** the blue-dragon bowl that, filled with sulphur-yellow roses, stood before him.*
- *tink can t go a twentieth part of the way **round**, she reminded him a little tartly.*
- *a large **round** rock, placed solidly on its base, was the only spot to which they seemed to lead.*

### 1900-1940

- *the ode ran thus:-- the warrior fights, and dies for fame-- the empty glories of a name;-- but we who linger **round** this spot, the warrior's guerdon covet nott.*
- *he swept his arm **round** the company inclusively.*
- *thanks, i ve turned **round** already.*

### 1940-1980

- *john campbell's hermippus redivivus , 85 and centres **round** the theories of the rosicrucians.*
- *no, they hold there a large, **round** wad of tow and cork, enveloped in the thickest and toughest of ox-hide.*
- *with a triumphant smile, they were told, that it was ten miles **round**.*
