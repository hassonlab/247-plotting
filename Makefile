
PDIR := $(shell dirname `pwd`)
USR := $(shell whoami | head -c 2)


######################################################################################
####################################  Setting up  ####################################
######################################################################################

# Sync electrode files (from /projects/HASSON/247/plotting)
# Link data files (from pickling/encoding/decoding results to data/)
# Create results folder for figures
link-data:
	mkdir -p data
	mkdir -p data/plotting
	rsync -rav /projects/HASSON/247/plotting/* data/plotting/
	ln -fs $(PDIR)/247-pickling/results data/pickling
	ln -fs $(PDIR)/247-encoding/results data/encoding
	mkdir -p results
	mkdir -p results/figures

# resync data from projects
sync-data:
	rsync -rav /projects/HASSON/247/plotting/* data/plotting/



######################################################################################
#####################################  Embedding  ####################################
######################################################################################


# layer index (use -1 for set layers)
LAYER_IDX := $(shell seq 0 4)
LAYER_IDX := -1

# whether to aggregate and average datum (comment out to not run this step)
AGGR := --aggregate

# whether to perform tsne (comment out to not run this step)
TSNE := --tsne

# whether to perform classification (comment out to not run this step)
CLASS := --classify

# Aggregate type (all words, 1st instance of words, average embeddings)
AGGR_TYPE := all
AGGR_TYPE := 1st
AGGR_TYPE := ave


CMD := echo
CMD := sbatch submit.sh
CMD := python


emb-class:
	$(CMD) scripts/tfsemb_class.py \
		$(AGGR) \
		$(TSNE) \
		$(CLASS) \
		--aggr-type $(AGGR_TYPE) \
		--savedir results/paper-whisper \
		--layer $(LAYER_IDX); \


emb-class-layers:
	for layer in $(LAYER_IDX); do \
		$(CMD) scripts/tfsemb_class.py \
			$(AGGR) \
			$(TSNE) \
			$(CLASS) \
			--aggr-type $(AGGR_TYPE) \
			--savedir results/paper-whisper \
			--layer $$layer; \
	done;


emb-class-new:
	$(CMD) scripts/tfsemb_class-preds.py



######################################################################################
#####################################  Encoding  #####################################
######################################################################################

# make sure the lags and the formats are in the same order
LAGS1 := {-10000..10000..25}
LAGS2 := -60000 -50000 -40000 -30000 -20000 20000 30000 40000 50000 60000
LAGS3 := -150000 -120000 -90000 90000 120000 150000
LAGS4 := -300000 -250000 -200000 200000 250000 300000
LAGS_FINAL := -300000 -60000 -30000 {-10000..10000..25} 30000 60000 300000 # final
# LAGS_FINAL := -99999999 # select all the lags that are concatenated (quardra)


concat-lags:
	python scripts/tfsenc_concat.py \
		--formats \
			'data/encoding/tfs//' \
		--lags \
			$(LAGS1) \
			$(LAGS2) \
			$(LAGS3) \
			$(LAGS4) \
		--lags-final $(LAGS_FINAL) \
		--output-dir data/encoding/tfs/concat/



######################################################################################
#####################################  Plotting  #####################################
######################################################################################


# For a more detailed explanation of the plotting arguments, look here: https://github.com/hassonlab/247-plotting/wiki/Encoding-Arguments

# LAGS_PLT: lags from encoding (should have the same lags as the data files from formats)
# LAGS_SHOW: lags to show in plot (lags that we want to plot, could be all or part of LAGS_PLT)

# X_VALS_SHOW: x-values for those lags we want to plot (same length as LAGS_SHOW) \
(for regular encoding, X_VALS_SHOW should be the same as LAGS_SHOW) \
(for concatenated lags, such as type Quardra and type Final plots, X_VALS_SHOW is different from LAGS_SHOW)

# LAG_TKS: lag ticks (tick marks to show on the x-axis) (optional)
# LAT_TK_LABLS: lag tick labels (tick mark lables to show on the x-axis) (optional)

LAGS_PLT := {-2000..2000..25} # lag2k-25
LAGS_PLT := {-10000..10000..25} # lag10k-25
LAGS_PLT := {-5000..5000..25} # lag5k-25

# Plotting for vanilla encoding (no concatenated lags)
LAGS_SHOW := $(LAGS_PLT)
X_VALS_SHOW := $(LAGS_SHOW)
LAG_TKS := 
LAG_TK_LABLS :=

# zoomed-in version (from -2s to 2s)
LAGS_SHOW := {-2000..2000..25}
X_VALS_SHOW := {-2000..2000..25}
LAG_TKS := 
LAG_TK_LABLS :=

# Line color by (Choose what lines colors are decided by) (required) (labels or keys)
# Line style by (Choose what line styles are decided by) (required) (labels or keys)
# Split Direction, if any (Choose how plots are split) (optional) (horizontal or vertical)
# Split by, if any (Choose how lines are split into plots) (Only effective when Split is not empty) (optional) (labels or keys)
PLT_PARAMS := --lc-by labels --ls-by keys # plot for just one key (podcast plots)
PLT_PARAMS := --lc-by labels --ls-by keys --split horizontal --split-by keys # plot for prod+comp (247 plots)

# y-axis limits (for individual plots) (leave it 0 for automatic)
Y_LIMIT := 0 0.3
Y_LIMIT := 0

# Figure Size (width height)
FIG_SZ:= 15 6
FIG_SZ:= 18 6

# Significant electrode file directory
SIG_FN_DIR := 'data/plotting/sig-elecs'
SIG_FN_DIR := 'data/plotting/sig-elecs/20230510-tfs-sig-file'
SIG_FN_DIR := 'data/plotting/sig-elecs/20230413-whisper-paper'
SIG_FN_DIR := 'data/plotting/sig-elecs/20230723-tfs-sig-file'

# Significant electrode files
SIG_FN := --sig-elec-file tfs-sig-file-%s-whisper-en-last-0.01-comp.csv tfs-sig-file-%s-whisper-de-best-0.01-prod.csv
SIG_FN := --sig-elec-file podcast_160.csv
SIG_FN := --sig-elec-file tfs-sig-file-%s-whisper-ende-outer-comp.csv tfs-sig-file-%s-whisper-ende-outer-prod.csv
SIG_FN := 
SIG_FN := --sig-elec-file tfs-sig-file-glove-%s-comp.csv tfs-sig-file-glove-%s-prod.csv


plot-encoding:
	rm -f results/figures/*
	python scripts/tfsplt_encoding.py \
		--sid 676 \
		--formats \
			'data/encoding/tfs/20230811-transcription-test/kw-tfs-full-676-gpt2-xl--lag5k-25-all/*/*_%s.csv' \
			'data/encoding/tfs/20230811-transcription-test/kw-tfs-full-676-gpt2-xl-large-v2-x--lag5k-25-all/*/*_%s.csv' \
		--labels gpt2 gpt2-whisperx \
		--keys comp prod \
		--sig-elec-file-dir $(SIG_FN_DIR)\
		$(SIG_FN) \
		--fig-size $(FIG_SZ) \
		--lags-plot $(LAGS_PLT) \
		--lags-show $(LAGS_SHOW) \
		--x-vals-show $(X_VALS_SHOW) \
		$(LAG_TKS) \
		$(LAG_TK_LABLS) \
		$(PLT_PARAMS) \
		--y-vals-limit $(Y_LIMIT) \
		--outfile results/figures/tfs-676-whisperx-gpt2.pdf
	rsync -av results/figures/ ~/tigress/247-encoding-results/


plot-encoding-layers:
	rm -f results/figures/*
	python scripts/tfsplt_encoding-layers.py \
		--sid 625 676 7170 798 \
		--formats \
			'data/encoding/tfs//*/*_%s.csv' \
		--labels $(shell seq 0 24) \
		--colors viridis \
		--keys comp prod \
		--sig-elec-file-dir $(SIG_FN_DIR)\
		$(SIG_FN) \
		--fig-size $(FIG_SZ) \
		--lags-plot $(LAGS_PLT) \
		--lags-show $(LAGS_SHOW) \
		--x-vals-show $(X_VALS_SHOW) \
		$(LAG_TKS) \
		$(LAG_TK_LABLS) \
		--y-vals-limit $(Y_LIMIT) \
		--outfile results/figures/tfs-encoding-layers.pdf
	rsync -av results/figures/ ~/tigress/247-encoding-results/


plot-brainmap:
	rm -f results/figures/*
	python scripts/tfsplt_brainmap.py \
		--sid 625 676 7170 798 \
		--formats \
			'/data/encoding/tfs//*/*_%s.csv' \
		--effect max \
		--keys comp prod \
		--lags-plot $(LAGS_PLT) \
		--lags-show $(LAGS_SHOW) \
		--sig-elec-file-dir $(SIG_FN_DIR) \
		$(SIG_FN) \
		--outfile fig_%s.png
	rsync -av results/figures/ ~/tigress/247-encoding-results/


plot-brainmap-subjects:
	rm -f results/figures/*
	python scripts/tfsplt_brainmap_cat.py \
		--sid 625 676 7170 798 \
		--keys comp prod \
		--sig-elec-file-dir $(SIG_FN_DIR) \
		$(SIG_FN) \
		--outfile fig_%s.png
	rsync -av results/figures/ ~/tigress/247-encoding-results/
