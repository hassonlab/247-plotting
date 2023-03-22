import pandas as pd

############################################################
## SCRIPT TO READ IN SIG ELECTRODES AND WRITE TO CSV FILE ##
############################################################
sid = 7170

# model = "whisper-en-last"
# model = "whisper-de-best"
model = "whisper-de-last"

# key = "prod"
# key = "comp"

electrode = ['LGA7', 'RAT2', 'LGA14', 'LGA22', 'LGA30', 'LGB91', 'LGB75', 'LGA13']

electrode2 =  ['LGA37', 'LGA10', 'LGA40']

electrode.extend(electrode2)
subject = [sid] * len(electrode)

out_path = f'/scratch/gpfs/ln1144/247-plotting/data/plotting/google-tfs-sig-files/tfs-sig-file-{sid}-{model}-prod-comp-contrast-0.01.csv'

data = {'subject': subject, 'electrode': electrode}

df = pd.DataFrame(data)

df.to_csv(out_path,header=True,index=False)

#############################################################
## IF THERE ARE MULTIPLE FILES THAT WE WANT TO CONCATENATE ##
#############################################################

# SIDS = [625, 625, 625, 625]
# KEYS = ['comp','prod']

# for sid in SIDS:
#     # for key in KEYS:
    
#         in_path1 = f'/scratch/gpfs/ln1144/247-plotting/data/plotting/google-tfs-sig-files/tfs-sig-file-{sid}-whisper-de-last-prod-comp-contrast-1.csv'
#         in_path2 = f'/scratch/gpfs/ln1144/247-plotting/data/plotting/google-tfs-sig-files/tfs-sig-file-{sid}-whisper-de-last-prod-comp-contrast-2.csv'

#         df1 = pd.read_csv(in_path1)
#         df2 = pd.read_csv(in_path2)

#         out_df = pd.concat([df1,df2])

#         out_path = f'/scratch/gpfs/ln1144/247-plotting/data/plotting//google-tfs-sig-files/tfs-sig-file-{sid}-whisper-de-last-prod-comp-contrast.csv'

#         out_df.to_csv(out_path, index=False)






