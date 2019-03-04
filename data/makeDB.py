import pandas as pd
import pyrosetta
from pyrosetta.toolbox import pose_from_rcsb
from pyrosetta.teaching import *
import argparse
pyrosetta.init()
import os.path
import csv
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("id_file", help="Text file containing all pdbs you wish to put in database")
parser.add_argument("curr_file", help="Other file to include in database (will not be overwritten unless is the same as outfile)")
parser.add_argument("out_file", help="File to write db to")
args = parser.parse_args()

pdb_ids_file = args.id_file #'cameo_ids.txt'
curr_data_file = args.curr_file #'cameo_scores.csv'
out_file = args.out_file #'cameo_scores.csv'

scorefxn = get_fa_scorefxn()

# Subscore Types from Rosetat
score_types = [fa_atr,
         fa_rep,
         fa_sol,
         fa_intra_rep,
         fa_intra_sol_xover4,
         lk_ball_wtd,
         fa_elec,
         pro_close,
         hbond_sr_bb,
         hbond_lr_bb,
         hbond_bb_sc,
         hbond_sc,
         dslf_fa13,
         omega,
         fa_dun,
         p_aa_pp,
         yhh_planarity,
         ref, rama_prepro]
column_names=['pdb_id', 'seq', 'total'] + [str(s)[10:] for s in score_types]
# TODO change to: column_names=['pdb_id','total', 'seq', 'phi','psi','chi'] + [str(s)[10:] for s in score_types]


# load previous dataset
if os.path.isfile(curr_data_file):
    print("reading")
    curr_db = pd.read_csv(curr_data_file, index_col=0)
else:

curr_db = pd.DataFrame(columns=column_names, dtype=float)
# Load Queries
with open(pdb_ids_file, 'r') as f:
    ids_to_query = set(f.read().splitlines())
    
# Query all pbd ids we have not querried yet.
ids_to_query = ids_to_query - set(curr_db['pdb_id'])



# Querry
i = 0
for pdbid in ids_to_query:
    i += 1
    if i % 50 == 0:
        print(i, " done")
        curr_db.to_csv(out_file)
    pose = pose_from_rcsb(pdbid)
    total = scorefxn(pose)
    sub_scores = [pose.energies().total_energies()[sub_score] for sub_score in score_types]
    temp_db = pd.DataFrame([[pdbid, pose.sequence(), total] + sub_scores], columns=column_names, index=[len(curr_db)])
    curr_db = pd.concat([curr_db, temp_db], axis=0)
    
# Write db to out file
curr_db.to_csv(out_file)
