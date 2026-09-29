# Comparison datasets on disk, 2026-09-29

Folder: `data set for comparison`. Counts from `scripts/eval/scan_comparison_datasets.py`, saved in `comparison_dataset_manifest.json`.

| folder | what it is | images or clouds | labels | aligned 6 views of one object |
| --- | --- | ---: | --- | --- |
| A Diverse Image Dataset for Bangladeshi Currency Recognition | Bangladeshi notes | 5073 images, 9 denomination folders (2 through 1000) | denomination | no |
| Bangla Money dataset Kaggle | Bangladeshi notes | 1971 images. Training class folders sum to 1638. Testing is 333 files named `0.jpg`, `1.jpg`, ... with no class folder | denomination on the training tree only | no |
| Large Scale BDT DB 2026 Kaggle | Bangladeshi coins and one demonetized-note class | 100000 images, 10000 in each of 10 folders | coin or note class name | no |
| NSTU-BDTAKA | Bangladeshi notes | 31986 images and 3111 label files. Recognition classes are denomination folders such as `1000_taka`. Filenames contain `.rf.`, which is an augmentation copy | denomination and detection boxes | no |
| ModelNet40 normal_resampled | non-currency 3D shapes | 12318 point-cloud `.txt` files, 40 classes, 10000 points of xyz plus normals. A `.vs` directory is editor metadata and is ignored | shape class | not photographs. Six turntable renders are built in `train_modelnet_vcds.py` |
| MVP-N | not in the folder | 0 |  |  |
| Hugging Face `Msun/modelnet40` | not downloaded again | local `normal_resampled` files were used |  |  |

No USD, EUR, INR, CNY, JPY, or GBP folder is in this comparison set.

Authenticity of genuine versus counterfeit is measured on JaalTaka. These extra Bangladeshi sets do not label genuine versus counterfeit, so a PRMVT authenticity number was not computed on them. Scoring a denomination id as if it were authenticity would mix two tasks.

The ModelNet view-count run is a separate classification task on rendered shapes. The standing test file is `results/vcds_modelnet/subset10_seed42_e40.json`. Shorter runs are kept beside it.
