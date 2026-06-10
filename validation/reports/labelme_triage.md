# LabelMe triage

Cases: 20. Median wall span-coverage recall: **0.55**.

| Case | Bucket | wall cov R | wall strict R | room R | pred/gt rooms | pred/gt walls | px/ft |
|------|--------|------------|---------------|--------|---------------|---------------|-------|
| labelme_fp_27 | missing_interior | 0.07 | 0.00 | 0.42 | 7/12 | 13/5 | 65.56 |
| labelme_fp_35_1 | sparse_gt | 0.17 | 0.00 | 0.50 | 6/2 | 52/1 | 37.92 |
| labelme_fp_36_2 | sparse_gt | 0.18 | 0.00 | 1.00 | 3/1 | 12/2 | 32.38 |
| labelme_fp_86_2 | missing_interior | 0.22 | 0.00 | 0.50 | 1/2 | 5/2 | 41.34 |
| labelme_fp_35_2 | sparse_gt | 0.22 | 0.00 | 0.50 | 12/2 | 45/1 | 37.4 |
| labelme_fp_25_1 | sparse_gt | 0.26 | 0.60 | 0.00 | 1/4 | 18/5 | 74.23 |
| labelme_fp_60 | sparse_gt | 0.32 | 0.07 | 0.31 | 15/32 | 58/14 | 85.52 |
| labelme_fp_19 | missing_interior | 0.36 | 0.09 | 0.03 | 10/75 | 33/54 | 49.86 |
| labelme_fp_10 | sparse_gt | 0.44 | 0.31 | 0.75 | 22/16 | 56/13 | 72.82 |
| labelme_fp_48_1 | sparse_gt | 0.51 | 0.33 | 0.00 | 2/2 | 12/3 | 49.01 |
| labelme_fp_31_2 | mixed | 0.58 | 0.46 | 0.46 | 22/37 | 45/24 | 53.18 |
| labelme_fp_87 | sparse_gt | 0.62 | 0.00 | 1.00 | 2/2 | 11/2 | 33.16 |
| labelme_fp_20 | mixed | 0.65 | 0.17 | 0.53 | 15/17 | 48/23 | 43.22 |
| labelme_fp_6_1 | sparse_gt | 0.65 | 0.60 | 1.00 | 13/9 | 49/5 | 64.52 |
| labelme_fp_71 | sparse_gt | 0.65 | 0.55 | 0.00 | 17/0 | 44/11 | 64.39 |
| labelme_fp_1 | sparse_gt | 0.72 | 0.31 | 0.36 | 19/22 | 50/13 | 55.83 |
| labelme_fp_7_2 | sparse_gt | 0.74 | 0.67 | 0.80 | 6/5 | 18/3 | 66.54 |
| labelme_fp_25_2 | sparse_gt | 0.78 | 0.00 | 0.00 | 2/2 | 32/1 | 33.66 |
| labelme_fp_66_2 | sparse_gt | 0.88 | 0.00 | 0.00 | 9/1 | 51/1 | 40.28 |
| labelme_fp_54_2 | sparse_gt | 0.98 | 1.00 | 0.33 | 2/3 | 17/1 | 42.21 |

## Bucket counts

- **sparse_gt**: 15
- **missing_interior**: 3
- **mixed**: 2

## Worst 5 (wall span-coverage recall)

- `labelme_fp_27` (missing_interior): cov R=0.07, tags=['missing_interior'], cal=['dpi_out_of_range']
- `labelme_fp_35_1` (sparse_gt): cov R=0.17, tags=['sparse_gt'], cal=['dpi_out_of_range']
- `labelme_fp_36_2` (sparse_gt): cov R=0.18, tags=['sparse_gt'], cal=['dpi_out_of_range']
- `labelme_fp_86_2` (missing_interior): cov R=0.22, tags=['missing_interior'], cal=['dpi_out_of_range']
- `labelme_fp_35_2` (sparse_gt): cov R=0.22, tags=['sparse_gt'], cal=['dpi_out_of_range']
