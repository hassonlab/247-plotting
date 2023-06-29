
PDIR := $(shell dirname `pwd`)
USR := $(shell whoami | head -c 2)

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
CMD := sbatch submit1.sh
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


########################## Regular Plotting Parameters ##########################
# LAGS_PLT: lags to plot (should have the same lags as the data files from formats)
# LAGS_SHOW: lags to show in plot (lags that we want to plot, could be all or part of LAGS_PLT)

# X_VALS_SHOW: x-values for those lags we want to plot (same length as LAGS_SHOW) \
(for regular encoding, X_VALS_SHOW should be the same as LAGS_SHOW) \
(for concatenated lags, such as type Quardra and type Final plots, X_VALS_SHOW is different from LAGS_SHOW)

# LAG_TKS: lag ticks (tick marks to show on the x-axis) (optional)
# LAT_TK_LABLS: lag tick labels (tick mark lables to show on the x-axis) (optional)

LAGS_PLT := {-5000..5000..25} # lag5k-25
LAGS_PLT := {-10000..10000..25} # lag10k-25
LAGS_PLT := {-2000..2000..25} # lag2k-25

# Plotting for vanilla encoding (no concatenated lags)
LAGS_SHOW := $(LAGS_PLT)
X_VALS_SHOW := $(LAGS_SHOW)
LAG_TKS := 
LAG_TK_LABLS :=

# Plotting for type Quardra (four different concatenated lags for 247)
# LAGS_PLT := {-300000..-150000..50000} -120000 -90000 {-60000..-20000..10000} {-10000..10000..25} {20000..60000..10000} 90000 120000 {150000..300000..50000}
# LAGS_SHOW := $(LAGS_PLT)
# X_VALS_SHOW := {-28000..-16000..2000} {-15000..-12000..1000} {-10000..10000..25} {12000..15000..1000} {16000..28000..2000}
# LAG_TKS := --lag-ticks {-28..28..2}
# LAG_TK_LABLS := --lag-tick-labels -300 -250 -200 -150 -120 -90 -60 -40 -20 {-10..10..2} 20 40 60 90 120 150 200 250 300

# Plotting for type Final (final plots for 247) 
# LAGS_PLT := {-300000..-150000..50000} -120000 -90000 {-60000..-20000..10000} {-10000..10000..25} {20000..60000..10000} 90000 120000 {150000..300000..50000}
# LAGS_SHOW := -300000 -60000 -30000 {-10000..10000..25} 30000 60000 300000
# X_VALS_SHOW := -16000 -14000 -12000 {-10000..10000..25} 12000 14000 16000
# LAG_TKS := --lag-ticks {-16..16..2}
# LAG_TK_LABLS := --lag-tick-labels -300 -60 -30 {-10..10..2} 30 60 300

# zoomed-in version (from -2s to 2s)
LAGS_SHOW := {-2000..2000..25}
X_VALS_SHOW := {-2000..2000..25}
LAG_TKS := 
LAG_TK_LABLS :=

########################## Other Plotting Parameters ##########################
# Line color by (Choose what lines colors are decided by) (required)
# { --lc-by labels | --lc-by keys }

# Line style by (Choose what line styles are decided by) (required)
# { --ls-by labels | --ls-by keys }

# Split Direction, if any (Choose how plots are split) (optional)
# {  | --split horizontal | --split vertical }

# Split by, if any (Choose how lines are split into plots) (Only effective when Split is not empty) (optional)
# {  | --split-by labels | --split-by keys }

PLT_PARAMS := --lc-by labels --ls-by keys # plot for just one key (podcast plots)
PLT_PARAMS := --lc-by labels --ls-by keys --split horizontal --split-by keys # plot for prod+comp (247 plots)

# y-axis limits (for individual plots) (leave it 0 for automatic)
Y_LIMIT := 0 0.3
Y_LIMIT := 0

# Figure Size (width height)
FIG_SZ:= 15 6
FIG_SZ:= 18 6

# Note: if lc_by = labels, order formats by: glove (blue), gpt2 (orange), bbot decoder (green), fourth label (red)


# Significant electrode file directory and significant electrode files
SIG_FN_DIR := 'data/plotting/sig-elecs/20230510-tfs-sig-file'

# Note: when providing sig elec files, provide them in the (sid keys) combination order \
For instance, if sid = 625 676, keys = comp prod \
sig elec files should be in this order: (625 comp)(625 prod)(676 comp)(676 prod) \
The number of sig elec files should also equal # of sid * # of keys
SIG_FN := 
SIG_FN := --sig-elec-file tfs-sig-file-glove-625-comp.csv tfs-sig-file-glove-625-prod.csv tfs-sig-file-glove-676-comp.csv tfs-sig-file-glove-676-prod.csv tfs-sig-file-glove-7170-comp.csv tfs-sig-file-glove-7170-prod.csv tfs-sig-file-glove-798-comp.csv tfs-sig-file-glove-798-prod.csv



plot-encoding:
	rm -f results/figures/*
	python scripts/tfsplt_encoding.py \
		--sid 625 676 7170 798 \
		--formats \
			'data/encoding/tfs/20230227-gpt2-preds/kw-tfs-full-625-glove50-lag10k-25-gpt2-xl-prob/*/*_%s.csv' \
			'data/encoding/tfs/20230227-gpt2-preds/kw-tfs-full-625-glove50-lag10k-25-gpt2-xl-improb/*/*_%s.csv' \
			'data/encoding/tfs/20230227-gpt2-preds/kw-tfs-full-676-glove50-lag10k-25-gpt2-xl-prob/*/*_%s.csv' \
			'data/encoding/tfs/20230227-gpt2-preds/kw-tfs-full-676-glove50-lag10k-25-gpt2-xl-improb/*/*_%s.csv' \
			'data/encoding/tfs/20230227-gpt2-preds/kw-tfs-full-7170-glove50-lag10k-25-gpt2-xl-prob/*/*_%s.csv' \
			'data/encoding/tfs/20230227-gpt2-preds/kw-tfs-full-7170-glove50-lag10k-25-gpt2-xl-improb/*/*_%s.csv' \
			'data/encoding/tfs/20230227-gpt2-preds/kw-tfs-full-798-glove50-lag10k-25-gpt2-xl-prob/*/*_%s.csv' \
			'data/encoding/tfs/20230227-gpt2-preds/kw-tfs-full-798-glove50-lag10k-25-gpt2-xl-improb/*/*_%s.csv' \
		--labels prob improb prob improb prob improb prob improb \
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
		--outfile results/figures/tfs-glove-probimprob.pdf
	rsync -av results/figures/ ~/tigress/247-encoding-results/


plot-encoding-twosplit:
	rm -f results/figures/*
	python scripts/tfsplt_encoding-twosplit.py \
		--sid 625 676 7170 798 \
		--formats \
			'data/encoding/tfs/stock-glove/kw-tfs-full-625-glove50-lag2k-25-all/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-625-glove50-lag2k-25-all/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-625-glove50-lag2k-25-all-concat-emb/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-625-glove50-lag2k-25-all-concat-emb/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-625-glove50-lag2k-25-all-concat-emb2/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-625-glove50-lag2k-25-all-concat-emb2/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-625-glove50-lag2k-25-all-concat-emb3/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-625-glove50-lag2k-25-all-concat-emb3/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-625-glove50-lag2k-25-all-concat-emb4/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-625-glove50-lag2k-25-all-concat-emb4/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-625-glove50-lag2k-25-all/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-625-glove50-lag2k-25-all/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-625-glove50-lag2k-25-all-concat-emb-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-625-glove50-lag2k-25-all-concat-emb-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-625-glove50-lag2k-25-all-concat-emb2-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-625-glove50-lag2k-25-all-concat-emb2-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-625-glove50-lag2k-25-all-concat-emb3-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-625-glove50-lag2k-25-all-concat-emb3-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-625-glove50-lag2k-25-all-concat-emb4-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-625-glove50-lag2k-25-all-concat-emb4-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-676-glove50-lag2k-25-all/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-676-glove50-lag2k-25-all/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-676-glove50-lag2k-25-all-concat-emb/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-676-glove50-lag2k-25-all-concat-emb/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-676-glove50-lag2k-25-all-concat-emb2/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-676-glove50-lag2k-25-all-concat-emb2/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-676-glove50-lag2k-25-all-concat-emb3/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-676-glove50-lag2k-25-all-concat-emb3/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-676-glove50-lag2k-25-all-concat-emb4/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-676-glove50-lag2k-25-all-concat-emb4/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-676-glove50-lag2k-25-all/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-676-glove50-lag2k-25-all/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-676-glove50-lag2k-25-all-concat-emb-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-676-glove50-lag2k-25-all-concat-emb-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-676-glove50-lag2k-25-all-concat-emb2-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-676-glove50-lag2k-25-all-concat-emb2-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-676-glove50-lag2k-25-all-concat-emb3-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-676-glove50-lag2k-25-all-concat-emb3-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-676-glove50-lag2k-25-all-concat-emb4-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-676-glove50-lag2k-25-all-concat-emb4-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-7170-glove50-lag2k-25-all/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-7170-glove50-lag2k-25-all/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-7170-glove50-lag2k-25-all-concat-emb/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-7170-glove50-lag2k-25-all-concat-emb/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-7170-glove50-lag2k-25-all-concat-emb2/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-7170-glove50-lag2k-25-all-concat-emb2/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-7170-glove50-lag2k-25-all-concat-emb3/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-7170-glove50-lag2k-25-all-concat-emb3/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-7170-glove50-lag2k-25-all-concat-emb4/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-7170-glove50-lag2k-25-all-concat-emb4/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-7170-glove50-lag2k-25-all/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-7170-glove50-lag2k-25-all/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-7170-glove50-lag2k-25-all-concat-emb-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-7170-glove50-lag2k-25-all-concat-emb-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-7170-glove50-lag2k-25-all-concat-emb2-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-7170-glove50-lag2k-25-all-concat-emb2-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-7170-glove50-lag2k-25-all-concat-emb3-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-7170-glove50-lag2k-25-all-concat-emb3-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-7170-glove50-lag2k-25-all-concat-emb4-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-7170-glove50-lag2k-25-all-concat-emb4-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-798-glove50-lag2k-25-all/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-798-glove50-lag2k-25-all/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-798-glove50-lag2k-25-all-concat-emb/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-798-glove50-lag2k-25-all-concat-emb/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-798-glove50-lag2k-25-all-concat-emb2/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-798-glove50-lag2k-25-all-concat-emb2/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-798-glove50-lag2k-25-all-concat-emb3/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-798-glove50-lag2k-25-all-concat-emb3/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-798-glove50-lag2k-25-all-concat-emb4/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-798-glove50-lag2k-25-all-concat-emb4/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-798-glove50-lag2k-25-all/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-798-glove50-lag2k-25-all/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-798-glove50-lag2k-25-all-concat-emb-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-798-glove50-lag2k-25-all-concat-emb-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-798-glove50-lag2k-25-all-concat-emb2-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-798-glove50-lag2k-25-all-concat-emb2-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-798-glove50-lag2k-25-all-concat-emb3-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-798-glove50-lag2k-25-all-concat-emb3-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-798-glove50-lag2k-25-all-concat-emb4-nopca/*/*_%s.csv' \
			'data/encoding/tfs/stock-glove/kw-tfs-full-798-glove50-lag2k-25-all-concat-emb4-nopca/*/*_%s.csv' \
		--keys \
			'glove-n comp pca' \
			'glove-n prod pca' \
			'glove-n+1 comp pca' \
			'glove-n+1 prod pca' \
			'glove-n+2 comp pca' \
			'glove-n+2 prod pca' \
			'glove-n+3 comp pca' \
			'glove-n+3 prod pca' \
			'glove-n+4 comp pca' \
			'glove-n+4 prod pca' \
			'glove-n comp no-pca' \
			'glove-n prod no-pca' \
			'glove-n+1 comp no-pca' \
			'glove-n+1 prod no-pca' \
			'glove-n+2 comp no-pca' \
			'glove-n+2 prod no-pca' \
			'glove-n+3 comp no-pca' \
			'glove-n+3 prod no-pca' \
			'glove-n+4 comp no-pca' \
			'glove-n+4 prod no-pca' \
			'glove-n comp pca' \
			'glove-n prod pca' \
			'glove-n+1 comp pca' \
			'glove-n+1 prod pca' \
			'glove-n+2 comp pca' \
			'glove-n+2 prod pca' \
			'glove-n+3 comp pca' \
			'glove-n+3 prod pca' \
			'glove-n+4 comp pca' \
			'glove-n+4 prod pca' \
			'glove-n comp no-pca' \
			'glove-n prod no-pca' \
			'glove-n+1 comp no-pca' \
			'glove-n+1 prod no-pca' \
			'glove-n+2 comp no-pca' \
			'glove-n+2 prod no-pca' \
			'glove-n+3 comp no-pca' \
			'glove-n+3 prod no-pca' \
			'glove-n+4 comp no-pca' \
			'glove-n+4 prod no-pca' \
			'glove-n comp pca' \
			'glove-n prod pca' \
			'glove-n+1 comp pca' \
			'glove-n+1 prod pca' \
			'glove-n+2 comp pca' \
			'glove-n+2 prod pca' \
			'glove-n+3 comp pca' \
			'glove-n+3 prod pca' \
			'glove-n+4 comp pca' \
			'glove-n+4 prod pca' \
			'glove-n comp no-pca' \
			'glove-n prod no-pca' \
			'glove-n+1 comp no-pca' \
			'glove-n+1 prod no-pca' \
			'glove-n+2 comp no-pca' \
			'glove-n+2 prod no-pca' \
			'glove-n+3 comp no-pca' \
			'glove-n+3 prod no-pca' \
			'glove-n+4 comp no-pca' \
			'glove-n+4 prod no-pca' \
			'glove-n comp pca' \
			'glove-n prod pca' \
			'glove-n+1 comp pca' \
			'glove-n+1 prod pca' \
			'glove-n+2 comp pca' \
			'glove-n+2 prod pca' \
			'glove-n+3 comp pca' \
			'glove-n+3 prod pca' \
			'glove-n+4 comp pca' \
			'glove-n+4 prod pca' \
			'glove-n comp no-pca' \
			'glove-n prod no-pca' \
			'glove-n+1 comp no-pca' \
			'glove-n+1 prod no-pca' \
			'glove-n+2 comp no-pca' \
			'glove-n+2 prod no-pca' \
			'glove-n+3 comp no-pca' \
			'glove-n+3 prod no-pca' \
			'glove-n+4 comp no-pca' \
			'glove-n+4 prod no-pca' \
		--sig-elec-file-dir $(SIG_FN_DIR)\
		$(SIG_FN) \
		--fig-size $(FIG_SZ) \
		--lags-plot $(LAGS_PLT) \
		--lags-show $(LAGS_SHOW) \
		--x-vals-show $(X_VALS_SHOW) \
		$(LAG_TKS) \
		$(LG_TK_LABLS) \
		--y-vals-limit $(Y_LIMIT) \
		--lc-by 0 \
		--ls-by 1 \
		--split-hor 1 \
		--split-ver 2 \
		--outfile results/figures/tfs-glove-concat-twosplit.pdf
	rsync -av results/figures/ ~/tigress/247-encoding-results/



# HAS_CTX := --has-ctx
# SIG_ELECS := --sig-elecs

LAYER_IDX := $(shell seq 0 25)

CONDS := all correct incorrect
CONDS := all


plot-encoding-layers:
	rm -f results/figures/*
	python scripts/tfsplt_encoding-layers.py \
		--sid 777 \
		--layer-num 24 \
		--top-dir data/encoding/podcast-old/20220516-eric-paper-replication/podcast-zaid-mwf\=0 \
		--modes comp \
		--conditions $(CONDS) \
		$(HAS_CTX) \
		$(SIG_ELECS) \
		--outfile results/figures/podcast-ericplots.pdf