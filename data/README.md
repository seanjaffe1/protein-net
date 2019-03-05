# Generating Data
`python makeDB.py --ids_file 'cameo_ids.txt' --curr_file 'cameo_scores1.csv' --out_file 'cameo_scores2.csv'`

This command will copy over the data in `cameo_scores1.csv` in to `cameo_scores2` and add data for all pbd_ids in `cameo_ids.txt` not in `cameo_scores1.csv` to `cameo_Scores2.csv`
If curr_file and out_file are the same, `makeDB.py` will just add data for pdb_ids in `cameo_ids.txt` and not in `cameo_scores1.csv` 
