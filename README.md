# 247-plotting

Plotting code for results of the other modules

Current to-dos in the Makefile

Will slowly clean up stuff



### Types of Plots

- Brain maps
  - `tfsplt_brainmap.ipynb` from encoding to brain coordinate files
  - `ntools_elec_plot_onebrain.m` one color brain plot
  - `ntools_elec_plot_onebrain2.m` two colors brain plot
  - To Do:
    - clean up MATLAB script  
    - effect vs rgb color
    - color spectrums (1 or 2)
    - separate color for non-sig elecs
    - max, min limits

- Encoding
  - `tfsplt_color.ipynb` testing colors
  - `tfsplt_main.ipynb` usual encoding
  - To Do:
    - usual encoding
      - no splits, or a horizontal/vertical split
      - electrode selection (sig-list or specific criterias)
      - color palletes
    - Eric style encoding
      - normalized version
      - small max correlation plots
      - small layer_idx with peak time plots
      - different color palletes
    - two split encoding (same as encoding)
    - ERP plots (same as encoding)
    - Max correlation heatmaps (with regard to layer_#, cxt_len)
    - (probably not needed) sig-test for electrodes (max, min, range gaussian mix)

- Pickling results in Python To Do:
  - Data analysis on signal
  - Data analysis on datum
    - word length / word gap
    - content words
    - cosine distance



### General file structure

- Miscellaneous scripts
  -  `tfsmis_sigtest.py` significant electrode tests
  -  `tfsmis_sigtest2.py` aggregates encoding results to one csv
  -  ~~`tfsmis_brain.py` appends encoding result to coordinate files~~
  -  `tfsplt_utils.py` reading in encoding results in parallel

- Plotting scripts in Python
  - `tfsplt_old.py` old encoding plots (not quite sure what this does)
  - `tfsplt_new.py` usual encoding plots
  - `tfsplt_layer.py` plot different layers Eric style (can do two splits?)
  - `tfsplt_newnew.py` two split encoding
  - `tfsmis_cosine_distance.py`
  - `tfsmis_content_words.py`
  - `tfsmis_brainembplot.py`

- Plotting scripts in MatLab
  - `ntools_elec_plot_onebrain.m` one color brainmap
  - `ntools_elec_plot_onebrain2.m` two color brainmap
  - `ntools_elec_plot_onebrain3.m` whatever Bobbi's cooking up



