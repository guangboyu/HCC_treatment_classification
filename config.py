import glob

data_paths = glob.glob("Data_remove_wk1/*/*/")
treatment_path = glob.glob("*treatment*processed.csv")
num_classes = 4