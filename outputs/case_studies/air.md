# Case Study: *air*

**Classification:** drift candidate

## Frequency Profile

| Period | Frequency (per million) | Raw Count |
|--------|------------------------:|----------:|
| 1820-1860 | 529.3 | 1226 |
| 1860-1900 | 487.9 | 675 |
| 1900-1940 | 542.6 | 479 |
| 1940-1980 | 622.7 | 546 |

## Semantic Drift Analysis

- **Overall centroid distance:** 0.0656
  *(95% CI: [0.0493, 0.0895])*
- **Average pairwise distance:** 0.4507
- **Neighbor Jaccard distance:** 0.636
- **Statistically significant drift:** Yes

### Drift Per Period Transition

| Transition | Centroid Distance | Avg Pairwise Distance |
|------------|------------------:|----------------------:|
| 1820-1860 -> 1860-1900 | 0.0229 | 0.4699 |
| 1860-1900 -> 1900-1940 | 0.0162 | 0.4259 |
| 1900-1940 -> 1940-1980 | 0.0048 | 0.4203 |

## Nearest Neighbors Per Period

These are the words used in similar contexts in each period. Shifts in this list directly evidence semantic drift.

**1820-1860:** `expression` (0.73), `sense` (0.72), `wind` (0.71), `tone` (0.71), `space` (0.71), `spirit` (0.71), `voice` (0.70), `feeling` (0.70), `influence` (0.69), `form` (0.69)

**1860-1900:** `wind` (0.75), `sky` (0.73), `space` (0.72), `sense` (0.71), `breath` (0.71), `spirit` (0.71), `tone` (0.69), `voice` (0.69), `expression` (0.69), `smoke` (0.68)

**1900-1940:** `wind` (0.76), `breath` (0.75), `sky` (0.73), `smoke` (0.72), `space` (0.72), `spirit` (0.71), `land` (0.70), `ground` (0.69), `sound` (0.69), `tone` (0.68)

**1940-1980:** `sky` (0.75), `space` (0.74), `breath` (0.73), `wind` (0.71), `smoke` (0.70), `spirit` (0.69), `ground` (0.69), `land` (0.68), `blow` (0.68), `tone` (0.67)

## Example Sentences By Period

Representative usages from the corpus, with target word in **bold**.

### 1820-1860

- *and then she stared at him with a bewildered **air**.*
- *at this word, jean valjean, who was dejected and seemed overwhelmed, raised his head with an **air** of stupefaction.*
- *as to the pretence of trying her native **air**, i look upon that as a mere excuse.*

### 1860-1900

- *once more, he took me by both hands and surveyed me with an **air** of admiring proprietorship: smoking with great complacency all the while.*
- *he scratched his head with a perplexed **air**, and said: well, that beats anything!*
- *already she was reeling in the **air**.*

### 1900-1940

- *the lamps were still burning redly in the murky **air** and, across the river, the palace of the four courts stood out menacingly against the heavy sky.*
- *i don t know how she manages it, here in the open **air**.*
- *a warm shock of **air** heat of mustard hanched on mr bloom s heart.*

### 1940-1980

- *then it darted a thousand feet straight up into the **air**; then spiralized downwards, and went eddying again round his head.*
- *they have a sharp, shrewish look, which i do not like at all; and in her **air** altogether, there is a self-sufficiency without fashion, which is intolerable.*
- *her passage through the darkening **air** or the verse with its black vowels and its opening sound, rich and lutelike?*
