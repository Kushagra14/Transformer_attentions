# Flash Attention Revision Notes

Covers: the causal "diagonal", the online softmax merge formula, why each row has its own scale factor, why both numerator and denominator are rescaled, what stays in SRAM, what `diag(...)` means, and how the GPU executes it.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| `Q_i` | block of Br query rows (Br x d) |
| `K_j`, `V_j` | block of Bc key / value rows (Bc x d) |
| `S_ij` | score tile `Q_i K_j^T / sqrt(d)` (Br x Bc) |
| `m_i` | running row max, vector of length Br |
| `l_i` | running softmax denominator, vector of length Br |
| `O_i` | running output block (Br x d) |
| `m~_ij` | row max of the current tile `S_ij` |
| `P~_ij` | `exp(S_ij - m~_ij)`, unnormalized probabilities of the tile |

Structure is the same as the matmul kernel: a query block is like one output block (one pid), and the loop over key/value blocks is like the K loop.

---

## 2. The causal diagonal (masking)

In decoder attention, query position `i` may only see keys at positions `j <= i`. In the score matrix (rows = queries, columns = keys) the allowed entries are the lower triangle including the diagonal.

Cut S into Br x Bc tiles. Each tile is one of three kinds:

| Tile position | Contents | Kernel action |
|---|---|---|
| Below the diagonal | all keys are earlier than all queries | compute normally, no mask |
| Above the diagonal | all keys are in the future | skip entirely (no load, no compute) |
| On the diagonal | part allowed, part future | compute, then mask element by element |

Why it matters:

- Skipping the upper tiles removes about half of the compute and K/V loads.
- Only diagonal tiles pay for the element-wise mask, which sets future scores to `-inf`. After `exp` they become 0, so they add nothing to `l` or `O`, and `-inf` never wins the max.
- Each query row always has at least one allowed key (itself), so no row is all `-inf` and there is no 0/0.
- Triton kernels usually split the loop into stages: one for full tiles, one for diagonal tiles.

Example: N = 6, Br = Bc = 2 gives a 3 x 3 grid of tiles. Six tiles are computed (3 below the diagonal, 3 on it) and 3 are skipped. A diagonal tile has the mask pattern `[[allowed, masked], [allowed, allowed]]`.

> This diagonal is unrelated to the `diag(...)` in the merge formula (section 4).

---

## 3. Why blocks at all

- The score matrix is N x N per head. For N = 4096 that is about 16 million scores, and it grows quadratically.
- Standard attention writes S to HBM, reads it back for softmax, writes P, reads it back for `P @ V`. This memory traffic is the bottleneck, not arithmetic.
- SRAM is only about 100 to 200 KB per SM, so S cannot sit there.
- Blocking keeps each tile on chip and never writes the N x N matrix. Memory drops from O(N^2) to O(N).
- The result is **exact** (up to floating-point rounding). FLOPs rise slightly because of rescaling, which is cheap compared with the memory saved.

The `1 / sqrt(d)` scale is applied inside each block, **before** the max and `exp`, because `exp(x / sqrt(d)) != exp(x) / sqrt(d)`. Many kernels fold it into Q once before the loop.

---

## 4. The merge formula (from the original FlashAttention paper)

```
m_new = max(m_i, m~_ij)
l_new = exp(m_i - m_new) * l_i + exp(m~_ij - m_new) * l~_ij

O_i <- diag(l_new)^-1 * ( diag(l_i) * exp(m_i - m_new) * O_i
                         + exp(m~_ij - m_new) * P~_ij * V_j )
```

Here `l~_ij` is the row sum of `P~_ij`.

Reading it in four steps:

1. **Un-normalize the old part.** The stored `O_i` was already divided by the old `l_i`. Multiplying by `l_i` turns it back into a plain sum.
2. **Move it to the new max.** Multiply by `exp(m_i - m_new)`.
3. **Add the new block.** `P~V` was computed relative to the tile's own max `m~`, so multiply it by `exp(m~ - m_new)` to put it on the same reference.
4. **Divide by `l_new`.** This restores a normalized output.

Why a reference change is needed: both parts are sums of exponentials, each computed relative to a different max. To add them they must share one reference, `m_new`.

---

## 5. Each row needs its own scale factor

Softmax is applied per query row, and each row has its own scores, so each row has its own `m` and `l`. The factors therefore differ by row.

Example, Br = 2. Old row maxes `m = [2, 5]`. The new tile's row maxes `m~ = [4, 3]`, so `m_new = [4, 5]`.

| | Old part factor `exp(m_old - m_new)` | New part factor `exp(m~ - m_new)` |
|---|---|---|
| query 1 | `exp(2-4) = 0.135` | `exp(4-4) = 1` |
| query 2 | `exp(5-5) = 1` | `exp(3-5) = 0.135` |

Whichever part had the smaller max is the one that gets scaled down. The two rows are mirror images, which is why a single number for the whole matrix cannot work.

---

## 6. Numerator and denominator both get rescaled

For one row, the old blocks contribute:

- numerator: `sum exp(s - m_old) * v`
- denominator: `sum exp(s - m_old)`

The factor `exp(m_old - m_new)` sits inside both sums, so both pick it up. The final output is their ratio. Rescaling only the numerator would mix two references and give a wrong answer.

### Worked example

One query row, d = 1, `V = [10, 20, 30, 40]`. Scores are `[1, 2]` in block 1 and `[3, 0]` in block 2.

**Block 1:** `m = 2`

- exps: `e^-1 = 0.368` and `1`
- `l = 1.368`
- numerator = `0.368*10 + 1*20 = 23.68`

**Block 2:** `m~ = 3`, so `m_new = 3`

- `P~ = [1, e^-3 = 0.0498]`, so `l~ = 1.0498`
- `P~V = 1*30 + 0.0498*40 = 31.99`
- old factor `exp(2-3) = 0.368`, new factor `exp(3-3) = 1`

**Merge:**

- numerator = `0.368*23.68 + 31.99 = 8.71 + 31.99 = 40.70`
- `l_new = 0.368*1.368 + 1.0498 = 0.503 + 1.0498 = 1.553`
- `O = 40.70 / 1.553 = 26.2`

**Direct check** (softmax over all four scores relative to 3): weights `0.135, 0.368, 1, 0.050` sum to `1.553`; weighted numerator `1.353 + 7.358 + 30 + 1.992 = 40.70`. The ratio is again `26.2`. Matches.

**If only the numerator were rescaled:** `O = 40.70 / (1.368 + 1.0498) = 16.8`, which is wrong.

---

## 7. What `diag(...)` means

`l` is a **vector** (one number per row) and `O` is a **matrix** (Br x d). Matrix notation has no symbol for "multiply row r of O by entry r of a vector", so the paper builds a matrix: `diag(l)` puts `l` on the diagonal and zeros elsewhere.

```
[a 0] [row 1 of O]   =   [a * row 1]
[0 b] [row 2 of O]       [b * row 2]
```

So `diag(l) * O` means "scale row r of O by `l_r`", and `diag(l)^-1` divides each row by its own `l_r`.

It is notation only. In code you write a broadcast multiply:

```python
O = O * factor[:, None]
```

`[:, None]` turns the vector into a column and stretches it across the columns, the same trick used in the pointer arithmetic of the matmul kernel.

---

## 8. What stays in SRAM

The overall computation is **not** stored. Only a small running state for one query block is held:

| Item | Shape | Role |
|---|---|---|
| `Q_i` | Br x d | loaded once, reused for every key block |
| `K_j`, `V_j` | Bc x d each | loaded one block at a time, then discarded |
| `S_ij` / `P~_ij` | Br x Bc | temporary tile, discarded after use |
| `O_i` (running) | Br x d | accumulates across key blocks |
| `m_i`, `l_i` | Br each | running max and denominator |

For Br = Bc = 128 and d = 64, `O_i` is 8192 values (about 32 KB in fp32). Doubling N from 4096 to 8192 changes only the number of loop iterations, not the state held. Old blocks' scores are no longer needed because `(m, l, O)` summarizes their whole contribution.

### Loop order

- **Original FlashAttention:** outer loop over key/value blocks, inner loop over query blocks. Each `O_i` is read from HBM, updated and written back every outer iteration. That is the "Write `O_i`" in the formula.
- **FlashAttention-2:** query block is the outer loop, key/value blocks the inner loop. `O_i`, `m_i`, `l_i` stay in registers for the whole inner loop and are written once at the end. `O` is kept unnormalized, rescaled only by `exp(m_old - m_new)`, and divided by `l` once at the end. The result is the same with fewer operations.

This matches the matmul kernel: the accumulator stays in registers for the whole K loop and is stored once with `tl.store`. The Triton flash attention tutorial follows the FA2 order.

---

## 9. How the GPU executes it

No diagonal matrix is built. A real `Br x Br` diagonal matrix times `O` would cost about `Br * Br * d` multiply-adds, almost all multiplying by zero. The broadcast multiply costs `Br * d`.

One iteration of the key-block loop:

| Step | Kind of operation | Runs on |
|---|---|---|
| `S = Q K^T` | matrix multiply | tensor cores |
| scale by `1/sqrt(d)`, row max | elementwise + reduction | regular cores |
| `exp`, row sum | elementwise + reduction | regular cores (`exp` on special-function units) |
| rescale `O` by the factor | broadcast multiply | regular cores |
| `O += P V` | matrix multiply | tensor cores |

The two matmuls are the heavy FLOPs. The middle steps are cheap per element but not free, because tensor cores are so fast that `exp` and rescaling take a visible share of the time. This is why FA2 reduces non-matmul work.

Parallelism:

- **Across rows: parallel.** Each row has its own `m`, `l` and factor.
- **Across query blocks: parallel**, each as its own pid, like output blocks in the matmul.
- **Across key blocks for one query block: sequential**, because block j+1 needs the running `(m, l, O)` from block j.

---

## 10. Why divide by sqrt(d)

**Assumption:** entries of q and k are independent, with mean 0 and variance 1.

1. One product term `q_i k_i` has mean 0 and variance `E[q_i^2] * E[k_i^2] = 1`.
2. The score `s = q_1 k_1 + ... + q_d k_d` is a sum of d independent terms, and variances of independent variables add, so `Var(s) = d` and the standard deviation is `sqrt(d)`.
3. For a constant c, `Var(s / c) = Var(s) / c^2`. With `c = sqrt(d)`, `Var(s / sqrt(d)) = 1`. Dividing by d itself would overshoot to variance `1/d`.

### Why variances of independent variables add

Let `a = X - mu_X`, `b = Y - mu_Y`. Then `Var(X+Y) = E[a^2] + 2E[ab] + E[b^2]`. For independent X and Y the cross term splits as `E[a] * E[b] = 0`, so `Var(X+Y) = Var(X) + Var(Y)`.

Independence matters: `Y = X` gives `Var(2X) = 4 Var(X)`, and `Y = -X` gives a sum that is always 0.

### Why we square

Variance is the average squared distance from the mean. Scores are positive and negative, so a plain average cancels to 0. Squaring makes +2 and -2 both count as far from the mean. It is preferred over absolute value because variances add.

### Worked examples

- **d = 2**, all random ±1 draws: scores `+2, 0, 0, -2`, average of `s^2` is `(4+0+0+4)/4 = 2 = d`. After dividing by `sqrt(2)` it is `1`.
- **d = 4**: 16 outcomes with `s = -4, -2, 0, 2, 4` occurring `1, 4, 6, 4, 1` times. Average of `s^2` is `64/16 = 4 = d`.
- **Hand-picked** `Q = [[+1,-1],[-1,+1]]`, `K^T = [[+1,+1],[-1,-1]]`: `S = [[2,2],[-2,-2]]`, average of `s^2` is `4`, and `2` after scaling. It starts above the theoretical 2 because the picked values are all extreme with no zeros. "Variance = d" describes the average over all random draws, not one sample, like a coin landing heads four times in a row. The reduction factor of d holds for any sample.

### Why it matters for softmax

Large gaps push softmax toward one-hot, and the gradient factor `p(1-p)` goes to about 0.

| Case | Scores | p(first) | p(1-p) |
|---|---|---|---|
| d = 2, unscaled | `[2, -2]` | 0.982 | 0.018 |
| d = 2, scaled | `[1.41, -1.41]` | 0.944 | 0.053 |
| d = 64, unscaled | `[8, 0]` | 0.9997 | 0.0003 |
| d = 64, scaled | `[1, 0]` | 0.73 | 0.2 |

Scaling gives about 3x more gradient at d = 2 and about 600x at d = 64, so it matters most for large head dimensions.

Caveat: trained models do not follow the independence assumption exactly, but `sqrt(d)` is still the standard choice because it keeps scores in a sensible range at any d.

---

## 11. One-line summaries

- **Blocks:** never build the N x N matrix; keep a small running state `(m, l, O)` per query block.
- **Merge:** un-normalize old, move to new max, add new block (also moved to new max), divide by new `l`.
- **Per-row factors:** each row has its own max and denominator, so each row gets its own factor.
- **Numerator and denominator:** both are rescaled, and the output is their ratio.
- **`diag`:** notation for "scale each row by its own number"; implemented as `factor[:, None] * O`.
- **Causal diagonal:** skip tiles above it, mask only tiles on it.
- **`1/sqrt(d)`:** the score variance grows like d, so dividing by `sqrt(d)` returns it to about 1 and keeps softmax gradients alive.

---

## 12. Exercises

1. For d = 256, what is `sqrt(d)` and the typical unscaled gap? Is softmax closer to one-hot than at d = 64?
2. Redo the merge in section 6 with the blocks in the opposite order (`[3, 0]` first, then `[1, 2]`). Which part gets shrunk, and do you still get 26.2?
3. In the FA2 version, at which step is `l` still rescaled?
4. For Br = 128 and d = 64, how many multiplications does the broadcast rescale do, and how many would a `Br x Br` diagonal matrix times `O` do?
5. For N = 6 with Br = Bc = 2, label each of the 9 tiles as full, skipped or diagonal.
