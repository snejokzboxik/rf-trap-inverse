# Export row FEM diagnostic: row index 2

- Sample ID: `594`
- FEM configuration: practical synthetic-data configuration (500 µm central mesh)
- Robust minima mode: yes
- Canonical transform: `FEM = [-W3, -W1, -W4, -W2]`
- Recomputed minima are sorted by `sort_points_by_polar_angle`.

## Input displacements (metres)

| electrode | dx | dy |
|---|---:|---:|
| W1 | -0.00040956698716 | -0.000495230258096 |
| W2 | 0.000137556903793 | -0.000355231534233 |
| W3 | 0.000392602196575 | 0.000212538810175 |
| W4 | 0.000276993223108 | 0.00019000132757 |

## Stored minima versus recomputed minima

| mapping | status | min1 error (µm) | min2 error (µm) | min3 error (µm) | mean (µm) | max (µm) |
|---|---|---:|---:|---:|---:|---:|
| canonical-[-W3,-W1,-W4,-W2] | ok | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| wrong-direct-W-as-F | ok | 1507.085178 | 1009.815789 | 1167.072714 | 1227.991227 | 1507.085178 |

## Stored minima (metres)

```text
[[ 0.00395296  0.00207656]
 [-0.00392706  0.00058885]
 [-0.00040162 -0.00286828]]
```

### canonical-[-W3,-W1,-W4,-W2]

FEM displacements (F1..F4):

```text
[[-0.0003926  -0.00021254]
 [ 0.00040957  0.00049523]
 [-0.00027699 -0.00019   ]
 [-0.00013756  0.00035523]]
```
Recomputed minima (sorted):

```text
[[ 0.00395296  0.00207656]
 [-0.00392706  0.00058885]
 [-0.00040162 -0.00286828]]
```

### wrong-direct-W-as-F

FEM displacements (F1..F4):

```text
[[-0.00040957 -0.00049523]
 [ 0.00013756 -0.00035523]
 [ 0.0003926   0.00021254]
 [ 0.00027699  0.00019   ]]
```
Recomputed minima (sorted):

```text
[[ 0.00438177  0.00063177]
 [-0.00489237  0.00088532]
 [ 0.00065296 -0.00236836]]
```
