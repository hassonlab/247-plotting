import glob
import os
import pandas as pd
from scipy.io import loadmat
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
import socket

# Arguments
###############################################################################################

###############################################################################################
# Required Arguments:
# id: List of subject(s) to plot
    # if 1 subject given, plot with T1 coordinates on patient specific brain
    # if >1 subject given, plot with MNI corrdinates on average brain
    # Ex.: ['625']
    # Ex.: ['625','676','717','798'] 

# effect_file: List of file name(s) containing the effect values to plot
    # Ex.: ["tfs_ave_whisper-en-last-0.01_prod_sig_withCoor.csv"]
    # Ex.: ["tfs_ave_whisper-en-last-whisper-de-best-contrast-0.01_prod_sig_withCoor_pos.csv",
    #           "tfs_ave_whisper-en-last-whisper-de-best-contrast-0.01_prod_sig_withCoor_neg.csv"]

# coor_in_effect_file: Plot using coordinates from effect file, 1/0 TODO: change this

# cbar_titles: List of title(s) for colorbar(s)
    # Ex.: ["correlation"]
    # Ex.: ["A", "B"]

# outname: File name for saving plot
    # Ex.: "test_figure.png"
    # Ex.: "test_figure.svg"
###############################################################################################

###############################################################################################
# Optional Arguments:
# cbar_max: Maximum value on colorbar
    # Default = max effect value from file(s) in 'effect_file'
    # NOTE: Effect value cannot be greater than cbar_max

# cbar_min: Minimum value on colorbar
    # Default = min effect value from files(s) in 'effect_file'
    # NOTE: Effect value cannot be less than cbar_min

# colorscales: Colorscale(s) used to color electrodes
    # Defulat = yellow to red
    # TODO: default for multiple colorbars
    # If 1 colorbar: List of position, rgb color vector pairs
    # If >1 colorbar: Dictionary containing lists of position, rgb color vector pairs
            # Position 0 (bottom) and 1 (top) required. 
            # Additional center values allow for multiple colorscales in one colobar.           
    # Ex: Define one colorbar from blue to white to red
        # {cbar_titles[0]:[[0,'rgb(0,0,255)'], [0.5,'rgb(255,255,255)'], [1,'rgb(255,0,0)']]}
    # Ex: Define 2 colorbars from light green to green and light purple to purple
        # {cbar_titles[0]:[[0,'rgb(255,248,240)'],[1,'rgb(255,0,0)']], 
        # cbar_titles[1]:[[0,'rgb(240,248,255)'],[1,'rgb(0,0,255)']]}
###############################################################################################

#TODO:
# The sizing is right for the average brain, but it doesn't translate to individual patient brains.
elec_type = "All" #elec_type = "G", "EG", "S", "D"
cbar_visibility = True

# Required arguments:
id = ['625']
effect_file = ["glove_aph_prod.csv"]
coor_in_effect_file = 0
cbar_titles = ['corr']
outname = "test_figure.png"

# Optional arguments:
cbar_max = None
cbar_min = None
colorscales = None

def set_args(id):

    host = socket.gethostname()
    if host == "della":
        print("Data path for host: della is not known. Please add it to the set_args function.")
        main_dir = ""
    elif host == "scotty":
        main_dir = "/mnt/cup/labs/hasson/ariel/MainDir/247/"
    else:
        main_dir = "/Volumes/hasson/ariel/MainDir/247/"
    

    if len(id) > 1:
        coor_type = "MNI"
    else:
        coor_type = "T1"

    return main_dir, coor_type

# loading brain surface plot
def load_surf(path,id):

    if len(id) > 1:
        file = glob.glob(os.path.join(path, "*.mat"))
    else:
        file = glob.glob(os.path.join(path, id[0], "NYUdownload", "NY" + id[0] + "*.mat"))
    
    # if one hemisphere
    if len(file) == 1:
        file = ''.join(file)
        surf1 = loadmat(file)
        surf2 = []
    # if both hemispheres
    elif len(file) == 2:
        file1 = ''.join(file[0])
        file2 = ''.join(file[1])
        surf1 = loadmat(file1)
        surf2 = loadmat(file2)

    return surf1, surf2

# reading electrode coordinates
def read_coor(path,id):

    df_coor = pd.DataFrame()
    for sid in id:
        sid_path = os.path.join(path, sid)
        file = os.path.join(sid_path, sid + "-electrode-coordinates.csv")
        df = pd.read_csv(file)
        df['subject'] = sid
        df_coor = df_coor.append(df)

    return df_coor

# plot 3D brain
def plot_brain(surf1, surf2):

    # surf["faces"] is an n x 3 matrix of indices into surf["coords"]; connectivity matrix
    # Subtract 1 from every index to convert MATLAB indexing to Python indexing
    surf1["faces"] = np.array([conn_idx - 1 for conn_idx in surf1["faces"]])

    # Plot 3D surfact plot of brain, colored according to depth
    fig = go.Figure()

    fig.add_trace(go.Mesh3d(x=surf1["coords"][:,0], y=surf1["coords"][:,1], z=surf1["coords"][:,2],
                     i=surf1["faces"][:,0], j=surf1["faces"][:,1], k=surf1["faces"][:,2],
                     color='rgb(175,175,175)'))
    
    # if both hemispheres
    if surf2:
        surf2["faces"] = np.array([conn_idx - 1 for conn_idx in surf2["faces"]])

        fig.add_trace(go.Mesh3d(x=surf2["coords"][:,0], y=surf2["coords"][:,1], z=surf2["coords"][:,2],
                          i=surf2["faces"][:,0], j=surf2["faces"][:,1], k=surf2["faces"][:,2],
                          color="rgb(175,175,175)"))

    fig.update_traces(lighting_ambient=0.3)
    return fig

# plot 3D electrodes
def plot_electrodes(elec_names,X,Y,Z,cbar_title,colorscale):

    r = 1.5
    fignew = go.Figure()
    for elecname,center_x,center_y,center_z in zip(elec_names,X,Y,Z):
        u, v = np.mgrid[0:2*np.pi:26j, 0:np.pi:26j]
        x = r * np.cos(u)*np.sin(v) + center_x
        y = r * np.sin(u)*np.sin(v) + center_y
        z = r * np.cos(v) + center_z

        fignew.add_trace(go.Surface(x=x,y=y,z=z,surfacecolor=np.ones(shape=z.shape),name=elecname,
                      legendgroup=cbar_title,colorscale=colorscale))
    
    return fignew

# Set min/max of colorbar
def scale_colorbar(fignew, df, cbar_min, cbar_max, cbar_title):

    if cbar_min is not None:
        cmin = cbar_min
    else:
        cmin = df["effect"].min()

    if cbar_max is not None:
        cmax = cbar_max
    else:
        cmax = df["effect"].max()
    fignew.update_traces(cmin=cmin,cmax=cmax,colorbar_title=cbar_title,
                         colorbar_title_font_size=40,colorbar_title_side='right')
    
    return fignew
    
# Color electrodes according to effect
def electrode_colors(fignew, df, subset):

    # Once max, min of colorbar is set, you can just use the value you want to plot (e.g. correlation) to determine the coloring,
    # must be in array the same shape as z data
    if subset > 0:
        fignew.update_traces(colorbar_x = 1 + 0.2*subset)
    for elec_idx in range(0,len(fignew.data)):
         effect = df["effect"][df.index[df["subject"]+df["name"] == fignew.data[elec_idx]["name"]]].tolist()
         fignew.data[elec_idx]["surfacecolor"] = fignew.data[elec_idx]["surfacecolor"] * effect

    return fignew

def update_properties(fig):

    # Left hemisphere
    # TODO: add camera for other views
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-1.5, y=0, z=0)
    )

    scene = dict(
        xaxis = dict(visible=False),
        yaxis = dict(visible=False),
        zaxis = dict(visible=False),
        aspectmode='auto'
    )

    fig.update_layout(scene_camera=camera,scene=scene)
    fig.update_traces(lighting_specular=0.4,colorbar_thickness=40,colorbar_tickfont_size=30,
                      lighting_roughness=0.4,lightposition=dict(x=0, y=0, z=100))

    return fig

def main(id,effect_file,cbar_titles,outname,cbar_min,cbar_max,colorscales,coor_in_effect_file):
    #id = sys.argv[1]
    #eff_file_name = sys.argv[2]

    main_dir, coor_type = set_args(id)
    path = os.path.join(main_dir,"ecog_coordinates")
    
    surf1, surf2 = load_surf(path, id)
    fig = plot_brain(surf1, surf2)

    if coor_in_effect_file == 0:
        df_coor = read_coor(path,id)

    for subset, cbar_title in enumerate(cbar_titles):
        
        if colorscales is None:
            colorscale = [[0,'rgb(255,0,0)'], [1,'rgb(255,255,0)']]
        else:
            colorscale = colorscales[cbar_title] 

        eff_file = os.path.join(main_dir + "results/brain_maps/effects/" + effect_file[subset])
        df_eff = pd.read_csv(eff_file)
        df_eff['subject'] = df_eff['subject'].astype("string")

        if 'MNI_X' in df_eff.columns:
            df_coor = df_eff
            fignew = plot_electrodes(df_coor['index'],df_coor[coor_type+"_X"],df_coor[coor_type+"_Y"],df_coor[coor_type+"_Z"],
                cbar_title,colorscale)
        else:
            # Filter electrodes to plot
            df_coor = df_coor[df_coor.name.isin(df_eff.name)]
            fignew = plot_electrodes(df_coor['subject'] + df_coor['name'],df_coor[coor_type+"_X"],df_coor[coor_type+"_Y"],df_coor[coor_type+"_Z"],
                cbar_title,colorscale)
            
        fignew = scale_colorbar(fignew, df_eff, cbar_min, cbar_max, cbar_title)
        fignew = electrode_colors(fignew, df_eff, subset)
        
        # Add electrode traces to main figure
        for trace in range(0,len(fignew.data)):
            fig.add_trace(fignew.data[trace])

    fig = update_properties(fig)

    fig.write_image(outname, scale=6, width=1200, height=1000)

    return

if __name__ == "__main__":
    main(id,effect_file,cbar_titles,outname,cbar_min,cbar_max,colorscales,coor_in_effect_file)