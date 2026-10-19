| family   | dataset                  | model         | reference             |   effect |   p_value |   p_holm |     p_bh |
|:---------|:-------------------------|:--------------|:----------------------|---------:|----------:|---------:|---------:|
| A-skill  | toy system, lead 1       | boosting      | linear                |   0.0893 |  0.435    | 1        | 0.58     |
| A-skill  | toy system, lead 1       | nn_gaussian   | linear                |  -0.2132 |  3.4e-10  | 3.06e-09 | 1.02e-09 |
| A-skill  | toy system, lead 1       | nn_shash      | linear                |  -0.4538 |  4.26e-36 | 5.12e-35 | 5.12e-35 |
| A-skill  | toy system, lead 1       | deep_ensemble | linear                |  -0.3234 |  5.67e-22 | 6.23e-21 | 3.4e-21  |
| A-skill  | toy system, lead 3       | boosting      | linear                |   0.1904 |  0.09     | 0.45     | 0.135    |
| A-skill  | toy system, lead 3       | nn_gaussian   | linear                |  -0.4045 |  6.58e-10 | 5.26e-09 | 1.58e-09 |
| A-skill  | toy system, lead 3       | nn_shash      | linear                |  -0.4324 |  5.01e-08 | 3.51e-07 | 1e-07    |
| A-skill  | toy system, lead 3       | deep_ensemble | linear                |  -0.4156 |  1.07e-10 | 1.07e-09 | 4.28e-10 |
| A-skill  | toy system, lead 6       | boosting      | linear                |   0.5842 |  8.37e-07 | 5.02e-06 | 1.44e-06 |
| A-skill  | toy system, lead 6       | nn_gaussian   | linear                |   0.0521 |  0.545    | 1        | 0.629    |
| A-skill  | toy system, lead 6       | nn_shash      | linear                |  -0.0069 |  0.948    | 1        | 0.948    |
| A-skill  | toy system, lead 6       | deep_ensemble | linear                |  -0.0486 |  0.577    | 1        | 0.629    |
| B-skill  | CESM2-LE test members    | linear        | climatology           |  -0.0656 |  2.7e-17  | 1.89e-16 | 6.75e-17 |
| B-skill  | CESM2-LE test members    | boosting      | climatology           |  -0.0615 |  3.5e-16  | 2.1e-15  | 6.99e-16 |
| B-skill  | CESM2-LE test members    | nn_gaussian   | climatology           |  -0.0669 |  2.31e-18 | 2.08e-17 | 1.15e-17 |
| B-skill  | CESM2-LE test members    | nn_shash      | climatology           |  -0.0658 |  5.96e-19 | 5.96e-18 | 5.96e-18 |
| B-skill  | CESM2-LE test members    | deep_ensemble | climatology           |  -0.0666 |  5.22e-18 | 4.18e-17 | 1.74e-17 |
| B-vs-lin | CESM2-LE test members    | boosting      | linear                |   0.0041 |  0.0828   | 0.579    | 0.29     |
| B-vs-lin | CESM2-LE test members    | nn_gaussian   | linear                |  -0.0013 |  0.415    | 1        | 0.474    |
| B-vs-lin | CESM2-LE test members    | nn_shash      | linear                |  -0.0002 |  0.925    | 1        | 0.925    |
| B-vs-lin | CESM2-LE test members    | deep_ensemble | linear                |  -0.001  |  0.405    | 1        | 0.474    |
| B-conf   | CESM2-LE test members    | boosting      | shifted spread        |  -0.0542 |  0.0493   | 0.0986   | 0.0563   |
| B-conf   | CESM2-LE test members    | nn_gaussian   | shifted spread        |  -0.0575 |  0.0141   | 0.0986   | 0.0262   |
| B-conf   | CESM2-LE test members    | nn_shash      | shifted spread        |  -0.058  |  0.0141   | 0.0986   | 0.0262   |
| B-conf   | CESM2-LE test members    | deep_ensemble | shifted spread        |  -0.0608 |  0.0141   | 0.0986   | 0.0262   |
| B-skill  | observations             | linear        | climatology_loo       |  -0.0301 |  0.117    | 0.198    | 0.117    |
| B-skill  | observations             | boosting      | climatology_loo       |  -0.044  |  0.0273   | 0.136    | 0.0455   |
| B-skill  | observations             | nn_gaussian   | climatology_loo       |  -0.0402 |  0.0495   | 0.198    | 0.0651   |
| B-skill  | observations             | nn_shash      | climatology_loo       |  -0.0378 |  0.0521   | 0.198    | 0.0651   |
| B-skill  | observations             | deep_ensemble | climatology_loo       |  -0.0374 |  0.0632   | 0.198    | 0.0702   |
| B-vs-lin | observations             | boosting      | linear                |  -0.0139 |  0.143    | 0.857    | 0.29     |
| B-vs-lin | observations             | nn_gaussian   | linear                |  -0.0101 |  0.0701   | 0.561    | 0.29     |
| B-vs-lin | observations             | nn_shash      | linear                |  -0.0077 |  0.268    | 1        | 0.428    |
| B-vs-lin | observations             | deep_ensemble | linear                |  -0.0073 |  0.145    | 0.857    | 0.29     |
| B-conf   | observations             | boosting      | shifted spread        |  -0.0419 |  0.303    | 0.303    | 0.303    |
| B-conf   | observations             | nn_gaussian   | shifted spread        |  -0.1774 |  0.0328   | 0.0986   | 0.0437   |
| B-conf   | observations             | nn_shash      | shifted spread        |  -0.1893 |  0.0164   | 0.0986   | 0.0262   |
| B-conf   | observations             | deep_ensemble | shifted spread        |  -0.2148 |  0.0082   | 0.0656   | 0.0262   |
| B-smyle  | observed years 1971-2021 | boosting      | SMYLE ensemble spread |   0.1949 |  0.093    | 0.186    | 0.112    |
| B-smyle  | observed years 1971-2021 | nn_gaussian   | SMYLE ensemble spread |   0.3488 |  0.0233   | 0.116    | 0.0465   |
| B-smyle  | observed years 1971-2021 | nn_shash      | SMYLE ensemble spread |   0.3322 |  0.0465   | 0.14     | 0.0698   |
| B-smyle  | observed years 1971-2021 | deep_ensemble | SMYLE ensemble spread |   0.326  |  0.0233   | 0.116    | 0.0465   |
| B-smyle  | observed years 1971-2021 | SMYLE         | shifted spread        |   0.0492 |  0.651    | 0.651    | 0.651    |
| B-smyle  | observed years 1971-2021 | SMYLE         | climatology_loo       |  -0.1007 |  0.0036   | 0.0216   | 0.0216   |