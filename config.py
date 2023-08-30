import glob

data_paths = glob.glob("Data_remove_wk1/*/*/")
treatment_path = glob.glob("*treatment*processed.csv")
outcome_path = glob.glob("*outcome*processed.csv")
num_classes = 4