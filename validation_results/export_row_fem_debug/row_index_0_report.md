# Export row FEM diagnostic: row index 0

- Sample ID: `124`
- FEM configuration: practical synthetic-data configuration (500 µm central mesh)
- Robust minima mode: yes
- Canonical transform: `FEM = [-W3, -W1, -W4, -W2]`
- Recomputed minima are sorted by `sort_points_by_polar_angle`.

## Input displacements (metres)

| electrode | dx | dy |
|---|---:|---:|
| W1 | -8.00517800463e-05 | -0.000122939349771 |
| W2 | -0.000375115866307 | 0.000298110970716 |
| W3 | 0.000246959736878 | -7.28592685878e-05 |
| W4 | 0.000476665677565 | -0.000257966056548 |

## Stored minima versus recomputed minima

| mapping | status | min1 error (µm) | min2 error (µm) | min3 error (µm) | mean (µm) | max (µm) |
|---|---|---:|---:|---:|---:|---:|
| canonical-[-W3,-W1,-W4,-W2] | ok | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| wrong-direct-W-as-F | ok | 1308.489446 | 1566.834174 | 1450.275225 | 1441.866282 | 1566.834174 |

## Stored minima (metres)

```text
[[ 0.00087649  0.00294802]
 [-0.00476373 -0.00034905]
 [ 0.00324289 -0.00209637]]
```

### canonical-[-W3,-W1,-W4,-W2]

FEM displacements (F1..F4):

```text
[[-2.46959737e-04  7.28592686e-05]
 [ 8.00517800e-05  1.22939350e-04]
 [-4.76665678e-04  2.57966057e-04]
 [ 3.75115866e-04 -2.98110971e-04]]
```
Recomputed minima (sorted):

```text
[[ 0.00087649  0.00294802]
 [-0.00476373 -0.00034905]
 [ 0.00324289 -0.00209637]]
```

### wrong-direct-W-as-F

FEM displacements (F1..F4):

```text
[[-8.00517800e-05 -1.22939350e-04]
 [-3.75115866e-04  2.98110971e-04]
 [ 2.46959737e-04 -7.28592686e-05]
 [ 4.76665678e-04 -2.57966057e-04]]
```
Recomputed minima (sorted):

```text
[[ 0.00050227  0.00420186]
 [-0.00321997 -0.00061694]
 [ 0.00245042 -0.00331098]]
```
