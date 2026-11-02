| metric | OLD_helmet_nohelmet-2 | n640_default | n640_traffic |
|---|---|---|---|
| imgsz | 512 | 640 | 640 |
| conf | [0.55, 0.45] | [0.45, 0.4] | [0.45, 0.5] |
| val_mAP50 | 0.5504 | 0.8136 | 0.8204 |
| test_mAP50 | 0.5831 | 0.7149 | 0.7669 |
| test_mAP50-95 | 0.2682 | 0.3648 | 0.3959 |
| helmet_P | 0.7791 | 0.7788 | 0.7802 |
| helmet_R | 0.6025 | 0.7981 | 0.7826 |
| helmet_F1 | 0.6795 | 0.7883 | 0.7814 |
| no_helmet_P | 0.672 | 0.735 | 0.8077 |
| no_helmet_R | 0.4884 | 0.5 | 0.4884 |
| no_helmet_F1 | 0.5657 | 0.5952 | 0.6087 |
| macro_F1 | 0.6226 | 0.6918 | 0.695 |
| false_NO_helmet(n/%) | 5 / 1.6% | 7 / 2.2% | 2 / 0.6% |
| false_helmet(n/%) | 12 / 7.0% | 14 / 8.1% | 17 / 9.9% |
| missed(h/nh) | 123/76 | 58/72 | 68/71 |
| spurious(h/nh) | 43/36 | 59/24 | 54/18 |
| bg_FP/img | 0.0189 | 0.0755 | 0.0377 |
| hard_mean_macroF1 | 0.53 | 0.596 | 0.64 |
| worst_condition | motion_blur(k15) (0.26) | motion_blur(k15) (0.41) | motion_blur(k15) (0.49) |
| recall_xs<16 | 0.243 | 0.321 | 0.379 |
| recall_s16-32 | 0.546 | 0.645 | 0.644 |
| recall_m32-64 | 0.642 | 0.753 | 0.731 |
| recall_l>=64 | 0.718 | 0.844 | 0.85 |
| RIDERS_ONLY helmet P/R | 0.7889/0.6574 (n=108) | 0.7456/0.787 (n=108) | 0.787/0.787 (n=108) |
| RIDERS_ONLY no_helmet P/R | 0.75/0.5217 (n=69) | 0.7347/0.5217 (n=69) | 0.8163/0.5797 (n=69) |
| RIDERS_ONLY macro_F1 | 0.6663 | 0.688 | 0.7325 |
| RIDERS_ONLY false_NO_helmet / false_helmet | 1 / 5 | 2 / 5 | 1 / 5 |
| PH_rider_recall(h/nh) | 0.057/0.047 | 0.065/0.014 | 0.052/0.0 |

### Extreme conditions (test set, synthetic) — macro-F1

| condition | OLD_helmet_nohelmet-2 | n640_default | n640_traffic |
|---|---|---|---|
| clean | 0.623 | 0.692 | 0.695 |
| night(dark+noise) | 0.535 | 0.501 | 0.637 |
| streetlight(orange, dark) | 0.537 | 0.520 | 0.664 |
| dusk(gamma1.8) | 0.614 | 0.660 | 0.689 |
| underexposed(x0.35) | 0.624 | 0.654 | 0.689 |
| overexposed(sun) | 0.514 | 0.617 | 0.656 |
| hard_shadow | 0.609 | 0.681 | 0.689 |
| motion_blur(k9) | 0.462 | 0.539 | 0.589 |
| motion_blur(k15) | 0.263 | 0.411 | 0.488 |
| defocus(gauss7) | 0.575 | 0.655 | 0.661 |
| jpeg_q20 | 0.596 | 0.637 | 0.668 |
| jpeg_q8 | 0.467 | 0.605 | 0.611 |
| sensor_noise(s=20) | 0.538 | 0.607 | 0.664 |
| lowres(x0.5) | 0.619 | 0.681 | 0.681 |
| lowres(x0.25) | 0.505 | 0.588 | 0.589 |
| rain(synthetic) | 0.622 | 0.674 | 0.684 |
| fog/haze | 0.534 | 0.626 | 0.682 |
| partial_occlusion(25-45%) | 0.397 | 0.481 | 0.535 |

### Real-condition subsets (clean test) — macro-F1

| subset | OLD_helmet_nohelmet-2 | n640_default | n640_traffic |
|---|---|---|---|
| dark(mean_luma<80) (n_h=39, n_nh=2) | 0.258 | 0.3514 | 0.3194 |
| sharp (n_h=316, n_nh=169) | 0.623 | 0.69 | 0.6922 |
| single(1 head) (n_h=68, n_nh=20) | 0.7171 | 0.7822 | 0.8206 |
| source:andrewmvd (n_h=95, n_nh=68) | 0.7642 | 0.7249 | 0.7724 |
| normal_light (n_h=283, n_nh=170) | 0.6344 | 0.6993 | 0.7073 |
| crowded(>=5 heads) (n_h=124, n_nh=80) | 0.535 | 0.6221 | 0.5873 |
| 2-4 heads (n_h=130, n_nh=72) | 0.6514 | 0.7184 | 0.7377 |
| source:anees (n_h=14, n_nh=15) | 0.7564 | 0.8717 | 0.8928 |
| blurry(lapvar<100) (n_h=6, n_nh=3) | 0.6 | 0.7879 | 0.8546 |
| source:idd_bg (n_h=0, n_nh=0) | 0.0 | 0.0 | 0.0 |
| source:india (n_h=213, n_nh=89) | 0.498 | 0.642 | 0.6026 |
| elevated_CCTV(india 036 cam) (n_h=42, n_nh=2) | 0.0454 | 0.3472 | 0.3521 |
