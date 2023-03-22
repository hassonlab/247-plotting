import pandas as pd
import os

path = "/scratch/gpfs/ln1144/247-plotting"
os.chdir(path)

SIDS = [625, 676, 7170, 798]
MODELS = ["whisper-en-last-0.05"]
KEYS = ["prod", "comp"]

SIG = True

AREAS = ['supramarginal', 'bankssts', 'parstriangularis', 'ctx-rh-insula', 'ctx-lh-caudalanteriorcingulate', 'ctx-rh-transversetemporal', 'parsorbitalis', 'ctx-lh-superiortemporal', 'caudalmiddlefrontal', 'Right-Cerebral-White-Matter', 'inferiortemporal', 'entorhinal', 'Right-Hippocampus', 'Left-Amygdala', 'lateraloccipital', 'cuneus', 'Left-Pallidum', 'inferiorparietal', 'cSTG', 'cMTG', 'ctx-lh-parstriangularis', 'rostralmiddlefrontal', 'ctx-rh-unknown', 'superiorparietal', 'parsopercularis', 'Left-choroid-plexus', 'superiorfrontal', 'ctx-lh-parahippocampal', 'ctx-lh-precentral', 'ctx-rh-middletemporal', 'rSTG', 'postcentral', 'Left-Hippocampus', 'rMTG', 'mSTG', 'Unknown', 'precuneus', 'Left-VentralDC', 'ctx-lh-middletemporal', 'mMTG', 'Left-Cerebral-White-Matter', 'Left-Inf-Lat-Vent', 'ctx-lh-insula', 'ctx-lh-fusiform', 'temporalpole', 'ctx-lh-transversetemporal', 'precentral', 'fusiform', 'ctx-lh-inferiortemporal', 'lingual', 'lateralorbitofrontal', 'Right-Putamen', 'Left-Putamen']

# MOTOR
Precentral = ['precentral', 'ctx-lh-precentral']

# SENSORY  / PARIETAL
Postcentral = ['postcentral', 'superiorparietal']

SM = ['precentral', 'ctx-lh-precentral','postcentral', 'superiorparietal']

ACC = ['ctx-lh-caudalanteriorcingulate'] 

SMG = ['supramarginal','inferiorparietal']

Inferiorparietal = ['inferiorparietal']
Precuneus = ['precuneus']

# OCCIPITAL
occipital = ['lateraloccipital', 'lingual','fusiform','ctx-lh-fusiform','cuneus']

# FRONTAL 
IFG = ['parsopercularis', 'parstriangularis']
parsopercularis = ['parsopercularis']
parstriangularis = ['parstriangularis']
OFC = ['parsorbitalis', 'lateralorbitofrontal']
parsorbitalis = ['parsorbitalis']
lateralorbitofrontal = ['lateralorbitofrontal']
PFC = ['caudalmiddlefrontal', 'rostralmiddlefrontal', 'superiorfrontal']
MFG = ['caudalmiddlefrontal', 'rostralmiddlefrontal']
SFG = ['superiorfrontal']

# TEMPORAL
TP = ['temporalpole']
ATL = ['temporalpole','inferiortemporal', 'ctx-lh-inferiortemporal']
MTG = ['rMTG', 'cMTG']
STG = ['mSTG','rSTG','cSTG','ctx-lh-superiortemporal']
HG = ['ctx-lh-transversetemporal','ctx-rh-transversetemporal']
STS = ['bankssts']

# DEEP 
HC = ['ctx-lh-parahippocampal', 'Right-Hippocampus', 'Left-Hippocampus', 'entorhinal']
Amygdala = ['Left-Amygdala']
BG = ['Right-Putamen', 'Left-Putamen', 'Left-Pallidum']
Insula = ['ctx-rh-insula', 'ctx-lh-insula']

ROIS = [IFG,PFC,OFC,ATL,STG,MTG,Precentral,Postcentral,SMG]
ROIS_str = ['IFG','PFC','OFC','ATL','STG','MTG','Precentral','Postcentral','SMG']


for model in MODELS:

    #HACK
    out_path = f"/scratch/gpfs/ln1144/247-plotting/data/plotting/ROI-tfs-sig-files/tfs-sig-files-{model}"

    if not os.path.exists(out_path):
        os.makedirs(out_path)


    for key in KEYS:    
        for sid in SIDS:

            # read elec name file
            elec_file = f"data/plotting/brainplot/{sid}_elecs.csv"
            elec_df = pd.read_csv(elec_file)

            if SIG:
                # reading ROI files and translate between electrode naming conventions
                sig_file = f"data/plotting/google-tfs-sig-files/tfs-sig-file-{sid}-{model}-{key}.csv"
        
                sig_df = pd.read_csv(sig_file)
                sig_df['elec'] = sig_df.electrode 

                for i in range(0,len(sig_df.electrode)):
                    sig_df.loc[i,('elec')] = elec_df.elec2.loc[elec_df.elec == sig_df.electrode.loc[i]].item()

            for i in range(0,len(ROIS)):

                roi = ROIS[i]
                roi_str = ROIS_str[i]

                electrode = []

                for area in AREAS:

                    if area in roi:

                        # read area files
                        area_file = f"data/{sid}_main_ROIs.txt"
                        area_df = pd.read_csv(area_file, delim_whitespace=True, names =['elec','2','3','4','5','area'])  

                        # get electrodes in area
                        areas = area_df.elec[area_df.area == area].tolist()

                        # filter sig electrodes
                        if SIG:
                            electrode.append(sig_df.electrode[sig_df.elec.isin(areas)].tolist())
                        else:
                            electrode.append(elec_df.elec[elec_df.elec2.isin(areas)].tolist())
                            
                electrode = [item for elem in electrode for item in elem]
                electrode = list(set(electrode)) 
                data = {'subject': sid, 'electrode': electrode}

                out_df = pd.DataFrame(data)
                out_df.dropna(axis=0,inplace=True)

                #HACK
                # fn = os.path.join(out_path,f"/tfs-sig-file-{sid}-{roi_str}-{model}-{key}.csv")   
                fn = out_path + f"/tfs-sig-file-{sid}-{roi_str}-{model}-{key}.csv"
                
                out_df.to_csv(fn,header=True,index=False)


