Some horrible results due to imbalance of labels. 

For small-batch-data with 30 papers: 
# Entity Type Distribution in NER Dataset

| Entity Type                        | Count |
|------------------------------------|-------|
| CellLine                           | 494   |
| Drug                               | 66    |
| Incubation Time (hour)             | 23    |
| Cell Density (cells/well)          | 4     |
| Drug treatment duration (hour/day) | 4     |
| Drug treatment duration (hour)     | 4     |
| Drug Concentration (mol/L)         | 3     |


For large-batch-data with 500 papers: 
# Entity Type Distribution in the Larger NER Dataset

| Entity Type                                                   | Count  |
|---------------------------------------------------------------|--------|
| CellLine                                                      | 12203  |
| Drug                                                          | 666    |
| Incubation Time (hour)                                        | 624    |
| Drug treatment duration (hour)                                | 184    |
| Drug Concentration (mol/L)                                    | 108    |
| Drug treatment duration (hour/day)                            | 107    |
| Cell Density (cells/well)                                     | 57     |
| Culture duration (Days/Weeks)                                 | 5      |
| Reagent/kit/staining solution Manufacturer and Catalog Number | 3      |

## NER Label Distribution from prediction trained with large-batch-data: 

| ID  | Label            | Count  |
|-----|------------------|--------|
| 0   | O                | 609268 |
| 6   | B-CellLine       | 14093  |
| 7   | I-CellLine       | 1727   |
| 1   | B-Drug           | 790    |
| 5   | I-Drug           | 468    |
| 10  | I-Incubation     | 344    |
| 9   | B-Incubation     | 271    |
| 2   | L)               | 223    |
| 3   | I-Cell           | 193    |
| 4   | well)            | 193    |
| 8   | day)             | 151    |
