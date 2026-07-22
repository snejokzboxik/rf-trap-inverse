# Export row FEM diagnostic: row index 1

- Sample ID: `330`
- FEM configuration: practical synthetic-data configuration (500 µm central mesh)
- Robust minima mode: yes
- Canonical transform: `FEM = [-W3, -W1, -W4, -W2]`
- Recomputed minima are sorted by `sort_points_by_polar_angle`.

## Input displacements (metres)

| electrode | dx | dy |
|---|---:|---:|
| W1 | 0.000138489673442 | -0.000218457201757 |
| W2 | 6.8241631616e-05 | 0.000178339329238 |
| W3 | 0.000456412955562 | -0.000209036850317 |
| W4 | 7.62970432285e-05 | -0.000176231456684 |

## Stored minima versus recomputed minima

| mapping | status | min1 error (µm) | min2 error (µm) | min3 error (µm) | mean (µm) | max (µm) |
|---|---|---:|---:|---:|---:|---:|
| canonical-[-W3,-W1,-W4,-W2] | ok | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| wrong-direct-W-as-F | ok | 739.767701 | 1276.531367 | 857.273433 | 957.857500 | 1276.531367 |

## Stored minima (metres)

```text
[[ 0.00164239  0.00216904]
 [-0.00335754  0.00118391]
 [ 0.00091424 -0.00325719]]
```

### canonical-[-W3,-W1,-W4,-W2]

FEM displacements (F1..F4):

```text
[[-4.56412956e-04  2.09036850e-04]
 [-1.38489673e-04  2.18457202e-04]
 [-7.62970432e-05  1.76231457e-04]
 [-6.82416316e-05 -1.78339329e-04]]
```
Recomputed minima (sorted):

```text
[[ 0.00164239  0.00216904]
 [-0.00335754  0.00118391]
 [ 0.00091424 -0.00325719]]
```

### wrong-direct-W-as-F

FEM displacements (F1..F4):

```text
[[ 1.38489673e-04 -2.18457202e-04]
 [ 6.82416316e-05  1.78339329e-04]
 [ 4.56412956e-04 -2.09036850e-04]
 [ 7.62970432e-05 -1.76231457e-04]]
```
Recomputed minima (sorted):

```text
[[ 0.00167016  0.00290828]
 [-0.002208    0.00062884]
 [ 0.00078598 -0.00410481]]
```
