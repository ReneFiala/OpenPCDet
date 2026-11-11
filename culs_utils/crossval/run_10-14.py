# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 22:12:16 2026

@author: Admin
"""

import pandas as pd
import cv_read, cv_folds, cv_sets, cv_vis
import matplotlib.pyplot as plt
import sys

OUT_PATH = "./crossval_csv/"
iterations = 1000

df = cv_read.read_excel("./infos/Annotation figure IDs.xlsx")
df = cv_read.replace_filenames(df, "./infos/common_database_double_checked_creation.txt")
df = cv_read.drop_samples(df, ["59-79-1_10-20230807T081832.pcd", "51-97-1_8-20220705T054818.pcd"])
df = cv_read.add_file(df, "./infos/CDI_index.csv", "file_name", "file", ["C_scene"])

df = cv_read.add_file(df, "./infos/neighborhood_scale_difficulty.csv", "file_name", "filename", ["neighborhood_scale_difficulty"])
df = cv_read.add_file(df, "./infos/SPOI_fixed.csv", "file_name", "pcd_file_name", ["SPOI"])

categories = ["SPOI_binned"]
df_spoi = cv_read.add_quantile_bins(df, "SPOI", bins=3)
df_spoi = cv_read.add_combinations(df_spoi, categories)
folds = cv_folds.search_for_folds_parallel(df_spoi, col_weights="ones", iterations=iterations, categories=categories)
cvs = cv_sets.create_nested_cv_sets(folds, method_name="ci-3b", col_filename="file_name")
cv_sets.save_cvs(cvs, OUT_PATH)

categories = ["SPOI_binned", "species"]
df_spoi = cv_read.add_quantile_bins(df, "SPOI", bins=3)
df_spoi = cv_read.add_combinations(df_spoi, categories)
folds = cv_folds.search_for_folds_parallel(df_spoi, col_weights="ones", iterations=iterations, categories=categories)
cvs = cv_sets.create_nested_cv_sets(folds, method_name="ci-3b_sp", col_filename="file_name")
cv_sets.save_cvs(cvs, OUT_PATH)

categories = ["neighborhood_scale_difficulty_binned"]
df_ai = cv_read.add_quantile_bins(df, "neighborhood_scale_difficulty", bins=3)
df_ai = cv_read.add_combinations(df_ai, categories)
folds = cv_folds.search_for_folds_parallel(df_ai, col_weights="ones", iterations=iterations, categories=categories)
cvs = cv_sets.create_nested_cv_sets(folds, method_name="neighbor-3b", col_filename="file_name")
cv_sets.save_cvs(cvs, OUT_PATH)

categories = ["neighborhood_scale_difficulty_binned", "species"]
df_ai = cv_read.add_quantile_bins(df, "neighborhood_scale_difficulty", bins=3)
df_ai = cv_read.add_combinations(df_ai, categories)
folds = cv_folds.search_for_folds_parallel(df_ai, col_weights="ones", iterations=iterations, categories=categories)
cvs = cv_sets.create_nested_cv_sets(folds, method_name="neighbor-3b_sp", col_filename="file_name")
cv_sets.save_cvs(cvs, OUT_PATH)
