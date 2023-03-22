import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "/scratch/gpfs/ln1144/247-plotting"
os.chdir(path)

############################
###### Core Arguments ######
############################

# PRJ_ID = "podcast"
PRJ_ID = "tfs"

#HACK
AGGREGATE = True
GET_RESULTS = False

# SIDS = [625] # for testing / 1 patient
SIDS = [625, 676, 7170, 798]

KEYS = ["prod", "comp"]

MODELS = ["whisper-en-last", "whisper-de-last"]

# COR_TYPE = "ind" # unique brain coordinate + brain map per patient
COR_TYPE = "ave" # average brain coordinates (for several patients)

##### Encoding Results Folder #####
FORMATS = []
for sid in SIDS:
    FORMATS.append(
        {
    "whisper-en-last" : f"/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-{sid}-lag10k-25-all-4/",
    "whisper-de-best" : f"/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-decoder/whisper-tiny.en-decoder-{sid}-lag10k-25-all-3/",
    "whisper-de-last" : f"/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-decoder/whisper-tiny.en-decoder-{sid}-lag10k-25-all-4/",
    # "whisper-full-last" : f"tfs/20230216-whisper-full/kw-tfs-full-{sid}-whisper-tiny.en-l4/*/",
    # "whisper-full-best" : f"tfs/20230216-whisper-full/kw-tfs-full-{sid}-whisper-tiny.en-l3/*/",
        }
    )

# Output directory name
# OUTPUT_DIR = "results/cor-tfs-area-diff-after"
# OUTPUT_DIR = "results/cor-tfs-max-diff"
# OUTPUT_DIR = "results/cor-tfs-area-diff-before"
# OUTPUT_DIR = "results/cor-tfs-max"
OUTPUT_DIR = "results/cor-tfs-max-diff-prod-comp"

# AREA lags (used for add_area)
LAGS = np.arange(-10000,10025,25)
AREA_START = -500
AREA_END = -100

INPUT_DIR = OUTPUT_DIR

# whether to use significance list
# SIG_ELECS = False
SIG_ELECS = True # only sig elecs

if "cor-tfs-max" in INPUT_DIR: # significance dict for max cor
    SIG_DICT = {
        # "whisper-en-last-0.05": "whisper-en-last-0.05",
        # "whisper-en-last-0.01": "whisper-en-last-0.01",
        # "whisper-de-best-0.05": "whisper-de-best-0.05",
        # "whisper-de-best-0.01": "whisper-de-best-0.01",
        # "whisper-de-last-0.05": "whisper-de-last-0.05",
        # "whisper-de-last-0.01": "whisper-de-last-0.01",
        # "whisper-en-de-best-contrast-0.05": "whisper-en-de-best-contrast-0.05",
        # "whisper-en-de-best-contrast-0.01": "whisper-en-de-best-contrast-0.01",
        # "whisper-en-de-last-contrast-0.05": "whisper-en-de-last-contrast-0.05",
        # "whisper-en-de-last-contrast-0.01": "whisper-en-de-last-contrast-0.01",
        "whisper-en-last-prod-comp-contrast-0.05": "whisper-en-last-prod-comp-contrast-0.05",
        "whisper-en-last-prod-comp-contrast-0.01": "whisper-en-last-prod-comp-contrast-0.01",
        "whisper-de-best-prod-comp-contrast-0.05": "whisper-de-best-prod-comp-contrast-0.05",
        "whisper-de-best-prod-comp-contrast-0.01": "whisper-de-best-prod-comp-contrast-0.01",
        "whisper-de-last-prod-comp-contrast-0.05": "whisper-de-last-prod-comp-contrast-0.05",
        "whisper-de-last-prod-comp-contrast-0.01": "whisper-de-last-prod-comp-contrast-0.01"
    }
elif "cor-tfs-area" in INPUT_DIR: # significance dict for area
    SIG_DICT = {}

def get_base_df(sid, cor, emb_key):

    # Get brain coordinate file
    coordinatefilename = f"data/plotting/brainplot/{sid}_{cor}.txt"

    data = pd.read_csv(coordinatefilename, sep=" ", header=None)
    data = data.set_index(0)
    data = data.loc[:, 1:4]
    print(f"\nFor subject {sid}:\ntxt has {len(data.index)} electrodes")

    # Get electrode name conversion file
    elecfilename = f"data/plotting/brainplot/{sid}_elecs.csv"
    elecs = pd.read_csv(elecfilename)
    elecs = elecs.dropna()
    elecs = elecs.rename(columns={"elec2": 0})
    elecs.set_index(0, inplace=True)

    df = pd.merge(data, elecs, left_index=True, right_index=True)
    print(f"Now subject has {len(df)} electrodes")
    
    # Create filler columns
    for col in emb_key:
        df[col] = -1

    return df

def read_file(filename, path):
    # Read in one electrode encoding correlation results
    filename = os.path.join("data/encoding/",path, filename)
    # breakpoint()
    if len(glob.glob(filename)) == 1:
        filename = glob.glob(filename)[0]
    elif len(glob.glob(filename)) == 0:
        return -1
    else:
        AssertionError("huh this shouldn't happen")
    elec_data = pd.read_csv(filename, header=None)
    return elec_data

def get_max(filename, path):
    
    # get max correlation for one electrode file
    fn = os.path.join(path,filename)

    #HACK
    try:
        elec_data = pd.read_csv(fn)
        max = elec_data.avg.max()
    except:
        max = -1
    
    return max

def get_area(filename, path, lags, chosen_lags):
    # get area under the curve for one electrode file
    elec_data = read_file(filename, path)
    if isinstance(elec_data, int):
        return -1
    elec_data = elec_data.loc[:, chosen_lags]
    x_vals = [lags[lag] / 1000 for lag in chosen_lags]

    return np.trapz(elec_data, x=x_vals, axis=1)  # integration

def add_encoding(df, sid, formats, type="max", lags = [], chosen_lags=[]):
    
    for format in formats:
        print(f"getting results for {format} embedding")
        for row, values in df.iterrows():
            col_name1 = format + "_prod"
            col_name2 = format + "_comp"
            #HACK
            #prod_name = f"{sid}_{values['elec']}_all_folds_prod.csv
            prod_name = f"{sid}_{values['elec']}_prod.csv"
            #HACK
            #comp_name = f"{sid}_{values['elec']}_all_folds_comp.csv"
            comp_name = f"{sid}_{values['elec']}_comp.csv"
            if type == "max":
                df.loc[row, col_name1] = get_max(prod_name, formats[format])
                df.loc[row, col_name2] = get_max(comp_name, formats[format])
            elif type == "area":
                df.loc[row, col_name1] = get_area(prod_name, formats[format], lags, chosen_lags)
                df.loc[row, col_name2] = get_area(comp_name, formats[format], lags, chosen_lags)
    return df

def get_area_diff(df, emb_key, mode="normalized"):
    for col in emb_key:
        if "incorrect" in col or "bot" in col: # incorrect column
            pass
        else: # correct column
            # get column names
            col2 = col.replace("correct", "incorrect") # incorrect column
            col2 = col2.replace("top", "bot") # bot column
            diff_col = col.replace("correct","")
            diff_col = diff_col.replace("top","")

            # normalized area diff
            df.loc[df[col] < 0, col] = 0  # turn negative area to 0
            df.loc[df[col2] < 0, col2] = 0  # turn negative area to 0
            if mode == "normalized": # normalized area diff
                df.loc[:,diff_col] = (df[col] - df[col2]) / df[[col, col2]].max(axis=1)
            elif mode == "normalized2": # area diff normalized
                df.loc[:,diff_col] = df[col] - df[col2]
                abs_max = max(abs(df.loc[:,diff_col].max()),abs(df.loc[:,diff_col].min()))
                df.loc[:,diff_col] = df.loc[:,diff_col] / abs_max
            elif mode == "none":
                df.loc[:,diff_col] = df[col] - df[col2]
            df.drop([col, col2], axis=1, inplace=True) # drop original columns

    return df

def save_file(df, sid, emb_keys, dir, cor, project):
    df.loc[:,0] = df.index

    for col in emb_keys:
        sid_file = os.path.join(dir, f"{sid}_{cor}_{col}.txt")
        # sids_file = os.path.join(dir, f"{project}_{cor}_{col}.txt")
        df_output = df.loc[:, [0, 1, 2, 3, 4, "elec", col]]
        df_output.dropna(inplace=True)
        df_output.to_csv(sid_file,index=False, header=False)
            
    return

def aggregate_results(input_dir, sids, cor_type, emb_name, key, sig_name):

    model1 = MODELS[0]
    model2 = MODELS[1]

    df_all = pd.DataFrame()
    
    for sid in sids:
        # load coordinate file
        if OUTPUT_DIR == "results/cor-tfs-max":
            model_name  = emb_name[:-5]
            cor_filename = os.path.join(input_dir,f"{sid}_{cor_type}_{model_name}_{key}.txt")
        elif OUTPUT_DIR == "results/cor-tfs-max-diff":
            cor_filename = os.path.join(input_dir,f"{sid}_{cor_type}_{model1}-{model2}-contrast_{key}.txt")
        elif OUTPUT_DIR == "results/cor-tfs-max-diff-prod-comp":
            model_name  = emb_name[:-5]
            cor_filename = os.path.join(input_dir,f"{sid}_{cor_type}_{model_name}.txt")
        
        df = pd.read_csv(cor_filename,names=["1","2","3","4","5","electrode","effect"])

        # load significance file
        if sig_name:
            if OUTPUT_DIR == "results/cor-tfs-max":
                sig_filename = os.path.join("data/plotting/google-tfs-sig-files",f"tfs-sig-file-{sid}-{sig_name}-{key}.csv")
            elif OUTPUT_DIR == "results/cor-tfs-max-diff":
                sig_filename = os.path.join("data/plotting/google-tfs-sig-files",f"tfs-sig-file-{sid}-{sig_name}-{key}.csv")
            elif OUTPUT_DIR == "results/cor-tfs-max-diff-prod-comp":
                sig_filename = os.path.join("data/plotting/google-tfs-sig-files",f"tfs-sig-file-{sid}-{sig_name}.csv")
  
            sig_df = pd.read_csv(sig_filename)
            df = pd.merge(df, sig_df, how='inner', left_on="electrode", right_on="electrode")
        
        # aggregate
        df_all = pd.concat([df_all,df])
        df_all.drop_duplicates(inplace=True)
        
    # save aggregate file
    df_output = df_all.loc[:, ["1","2","3","4","5","effect"]]
    sig_str = "_sig"
    if sig_name is None:
        sig_str = ""
    if OUTPUT_DIR == "results/cor-tfs-max":
        aggre_filename = os.path.join(input_dir,f"tfs_{cor_type}_{emb_name}_{key}{sig_str}.txt")
    elif OUTPUT_DIR == "results/cor-tfs-max-diff":
        sig_id = sig_name[-4:]
        aggre_filename = os.path.join(input_dir,f"tfs_{cor_type}_{model1}-{model2}-contrast-{sig_id}_{key}{sig_str}.txt")
    elif OUTPUT_DIR == "results/cor-tfs-max-diff-prod-comp":
        aggre_filename = os.path.join(input_dir,f"tfs_{cor_type}_{emb_name}{sig_str}.txt")

    df_output.to_csv(aggre_filename, index=False, header=False)

    return

#############################
## GET RESULTS PER PATIENT ##
#############################

if GET_RESULTS:
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    ##### Max correlation #####
    if OUTPUT_DIR == "results/cor-tfs-max":
        for sid, format in zip(SIDS, FORMATS):
            emb_key = [emb + "_" + key for emb in format.keys() for key in KEYS]
            df = get_base_df(sid, COR_TYPE, emb_key) # get all electrodes
            df = add_encoding(df, sid, format, "max") # add on the columns from encoding results
            save_file(df, sid, emb_key, OUTPUT_DIR, COR_TYPE, PRJ_ID) # save txt files

    ##### Difference between max correlation #####
    elif OUTPUT_DIR == "results/cor-tfs-max-diff":
        for sid in SIDS:
            for key in KEYS:

                model1 = MODELS[0]
                model2 = MODELS[1]
                
                f1 = f"results/cor-tfs-max/{sid}_ave_{model1}_{key}.txt"
                f2 = f"results/cor-tfs-max/{sid}_ave_{model2}_{key}.txt"

                df_1 = pd.read_csv(f1,names=["electrode","1","2","3","4","5","effect"]) # read txt files with max correlation for two models
                df_2 = pd.read_csv(f2,names=["electrode","1","2","3","4","5","effect"])

                assert(len(df_1) == len(df_2))
                df = df_1
                df.effect = df_1.effect - df_2.effect # get difference
                out_dir = f"results/cor-tfs-max-diff/{sid}_ave_{model1}-{model2}-contrast_{key}.txt"

                df.to_csv(out_dir ,index=False, header=False)

    #### Difference between prod and comp ####
    elif OUTPUT_DIR == "results/cor-tfs-max-diff-prod-comp":

        for sid in SIDS:
            for format in FORMATS:

                emb_keys = [emb for emb in format.keys()]
                
                for emb_key in emb_keys:

                    f1 = f"results/cor-tfs-max/{sid}_ave_{emb_key}_comp.txt"
                    f2 = f"results/cor-tfs-max/{sid}_ave_{emb_key}_prod.txt"

                    df_1 = pd.read_csv(f1,names=["electrode","1","2","3","4","5","effect"]) # read txt files with max correlation for two models
                    df_2 = pd.read_csv(f2,names=["electrode","1","2","3","4","5","effect"])

                    assert(len(df_1) == len(df_2))
                    df = df_1
                    df.effect = df_2.effect - df_1.effect # get difference

                    out_dir = f"results/cor-tfs-max-diff-prod-comp/{sid}_ave_{emb_key}-prod-comp-contrast.txt"

                    df.to_csv(out_dir ,index=False, header=False)

    ##### Difference in area under the curve #####
    elif "cor-tfs-area" in OUTPUT_DIR:
        chosen_lag_idx = [
            idx for idx, element in enumerate(LAGS) if (element >= AREA_START) & (element <= AREA_END)
        ] # calculate the correct lag idx

        for sid, format in zip(SIDS, FORMATS):
            emb_key = [emb + "_" + key for emb in format.keys() for key in KEYS]
            df = get_base_df(sid, COR_TYPE, emb_key) # get all electrodes
            df = add_encoding(df, sid, format, "area", LAGS, chosen_lag_idx) # add on columns from encoding results
            df = get_area_diff(df, emb_key, "normalized2") # get area difference

            # save txt files
            new_emb_key = [col.replace("incorrect","").replace("bot", "") for col in emb_key if "incorrect" in col or "bot" in col]
            save_file(df, sid, new_emb_key, OUTPUT_DIR, COR_TYPE, PRJ_ID) # save txt files


####################################################################
# AGGREGATE BRAIN COORDINATE FILES FOR ALL PATIENTS AND SIG ELECS ##
####################################################################

if AGGREGATE:
    for emb in SIG_DICT.keys():
        for key in KEYS:
            if SIG_ELECS:
                sig_name = SIG_DICT[emb]
            else:
                sig_name = None
            aggregate_results(INPUT_DIR, SIDS, COR_TYPE, emb, key, sig_name)


