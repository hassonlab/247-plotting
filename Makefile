# TODO

# create folders for datafiles on /projects
# include: coordinate files, sig-files, elec-name transition files (between python and matlab)

# commands to link data files (from pickling/encoding/decoding results to data/)
# commands to link data files (from /projects)

# commands for plotting


# clean up plotting scripts (find the paralleled ones)

######################### commands from encoding Makefile #########################

# Non-configurable paramters. Don't touch.
FILE := tfsenc_main
USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d-%H%M")
DT := ${USR}

# -----------------------------------------------------------------------------
#  Configurable options
# -----------------------------------------------------------------------------

PRJCT_ID := tfs
# {podcast | tfs}

# Sig file will override whatever electrodes you choose
SIG_FN := 
SIG_FN := --sig-elec-file tfs-sig-file-625-Precentral-$$model-comp.csv tfs-sig-file-676-Precentral-$$model-comp.csv tfs-sig-file-798-Precentral-$$model-comp.csv tfs-sig-file-7170-Precentral-$$model-comp.csv
# SIG_FN := --sig-elec-file test.csv
# SIG_FN := --sig-elec-file 129-phase-5000-sig-elec-glove50d-perElec-FDR-01-LH.csv
# SIG_FN := --sig-elec-file colton625.csv colton625.csv
# SIG_FN := --sig-elec-file tfs-sig-file-625-top-0.3-prod.csv tfs-sig-file-625-sig-0.3-comp.csv
# SIG_FN := --sig-elec-file 625-mariano-prod-new-53.csv 625-mariano-comp-new-30.csv # for sig-test
# SIG_FN := --sig-elec-file 676-mariano-prod-new-109.csv 676-mariano-comp-new-104.csv # for sig-test
# SIG_FN := --sig-elec-file 7170-comp-sig.csv 7170-prod-sig.csv
# SIG_FN := --sig-elec-file tfs-sig-file-676-sig-1.0-comp.csv tfs-sig-file-676-sig-1.0-prod.csv
# SIG_FN := --sig-elec-file tfs-sig-file-676-max-0.1-comp.csv tfs-sig-file-676-max-0.1-prod.csv
# SIG_FN := --sig-elec-file tfs-sig-file-region-parstri-prod.csv

### podcast significant electrode list (if provided, override electrode IDs)
# SIG_FN := --sig-elec-file podcast_160.csv

# Choose the lags to run for.
# LAGS := {400000..500000..100} # lag400500k-100
# LAGS := {-150000..150000..100} # lag60k-1k
# LAGS := {-500..500..5} # lag500-5
# LAGS := -300000 -250000 -200000 200000 250000 300000 # lag300k-50k
# LAGS := -150000 -120000 -90000 90000 120000 150000 # lag150k-30k
# LAGS := -60000 -50000 -40000 -30000 -20000 20000 30000 40000 50000 60000 # lag60k-10k
# LAGS := {-2000..2000..25} # lag2k-25
LAGS := {-10000..10000..25} # lag10k-25
 

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

########################## Regular Plotting Parameters ##########################
# LAGS_PLT: lags to plot (should have the same lags as the data files from formats)
# LAGS_SHOW: lags to show in plot (lags that we want to plot, could be all or part of LAGS_PLT)

# X_VALS_SHOW: x-values for those lags we want to plot (same length as LAGS_SHOW) \
(for regular encoding, X_VALS_SHOW should be the same as LAGS_SHOW) \
(for concatenated lags, such as type Quardra and type Final plots, X_VALS_SHOW is different from LAGS_SHOW)

# LAG_TKS: lag ticks (tick marks to show on the x-axis) (optional)
# LAT_TK_LABLS: lag tick labels (tick mark lables to show on the x-axis) (optional)

# Plotting for vanilla encoding (no concatenated lags)
LAGS_PLT := $(LAGS)
LAGS_SHOW := $(LAGS)
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
LAGS_PLT := $(LAGS)
LAGS_SHOW := {-2000..2000..25}
X_VALS_SHOW := $(LAGS_SHOW)
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

PLT_PARAMS := --ls-by labels --lc-by keys # plot for just one key (podcast plots)
# PLT_PARAMS := --lc-by labels --ls-by keys --split horizontal --split-by keys # plot for prod+comp (247 plots)

# Figure Size (width height)
FIG_SZ:= 15 12

ROI = Precentral
KEY = comp

# PARAMS FOR PLOTTING ROIS FOR DIFFERENT MODELS (SIG_FILES)
MODELS := whisper-en-last-0.05
KEYS := prod comp
ROIS := IFG PFC OFC ATL STG MTG Precentral Postcentral SMG

# folder name extensions
FN := -w12

# Note: if lc_by = labels, order formats by: glove (blue), gpt2 (orange), bbot decoder (green), fourth label (red)
# Note: when providing sig elec files, provide them in the (sid keys) combination order \
For instance, if sid = 625 676, keys = prod comp \
sig elec files should be in this order: (625 prod)(625 comp)(676 prod)(676 comp) \
The number of sig elec files should also equal # of sid * # of keys

# plot-whisper-rois:
# 	for model in $(MODELS); do \
# 		for roi in $(ROIS); do \
# 			for key in $(KEYS); do \
# 				python scripts/tfsplt_whisper.py \
# 					--sid 625 676 798 7170 \
# 					--formats \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder-sh-samples/whisper-tiny.en-encoder-sh-samples-625-lag10k-25-all-4/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder-sh-phonemes/whisper-tiny.en-encoder-sh-phonemes-625-lag10k-25-all-4/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder-sh-words-7.5/whisper-tiny.en-encoder-sh-words-7.5-625-lag10k-25-all-4/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-625-lag10k-25-all-4/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder-sh-samples/whisper-tiny.en-encoder-sh-samples-676-lag10k-25-all-4/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder-sh-phonemes/whisper-tiny.en-encoder-sh-phonemes-676-lag10k-25-all-4/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder-sh-words-7.5/whisper-tiny.en-encoder-sh-words-7.5-676-lag10k-25-all-4/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-676-lag10k-25-all-4/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder-sh-samples/whisper-tiny.en-encoder-sh-samples-798-lag10k-25-all-4/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder-sh-phonemes/whisper-tiny.en-encoder-sh-phonemes-798-lag10k-25-all-4/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder-sh-words-7.5/whisper-tiny.en-encoder-sh-words-7.5-798-lag10k-25-all-4/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-798-lag10k-25-all-4/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder-sh-samples/whisper-tiny.en-encoder-sh-samples-7170-lag10k-25-all-4/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder-sh-phonemes/whisper-tiny.en-encoder-sh-phonemes-7170-lag10k-25-all-4/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder-sh-words-7.5/whisper-tiny.en-encoder-sh-words-7.5-7170-lag10k-25-all-4/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-7170-lag10k-25-all-4/*_%s.csv' \
# 					--labels shuffle-samples shuffle-phonemes shuffle-words full-context-l4 shuffle-samples shuffle-phonemes shuffle-words full-context-l4 shuffle-samples shuffle-phonemes shuffle-words full-context-l4 shuffle-samples shuffle-phonemes shuffle-words full-context-l4 \
# 					--model $$model \
# 					--keys $$key \
# 					--roi $$roi\
# 					--sig-elec-file tfs-sig-file-625-$$roi-$$model-$$key.csv tfs-sig-file-676-$$roi-$$model-$$key.csv tfs-sig-file-798-$$roi-$$model-$$key.csv tfs-sig-file-7170-$$roi-$$model-$$key.csv  \
# 					--fig-size $(FIG_SZ) \
# 					--lags-plot $(LAGS_PLT) \
# 					--lags-show $(LAGS_SHOW) \
# 					--x-vals-show $(X_VALS_SHOW) \
# 					$(LAG_TKS) \
# 					$(LAG_TK_LABLS) \
# 					$(PLT_PARAMS) \
# 					--outfile results/figures/tfs-$$model-context-analysis-rois/tfs-$$roi-$$model-$$key.pdf; \
# 			done; \
# 		done; \
# 	done;

plot-whisper-rois:
	for model in $(MODELS); do \
		for roi in $(ROIS); do \
			for key in $(KEYS); do \
				python scripts/tfsplt_whisper.py \
					--sid 625 676 798 7170 \
					--formats \
						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder-w12/whisper-tiny.en-encoder-w12-625-lag10k-25-all-4/*_%s.csv' \
						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-625-lag10k-25-all-4/*_%s.csv' \
						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder-w12/whisper-tiny.en-encoder-w12-676-lag10k-25-all-4/*_%s.csv' \
						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-676-lag10k-25-all-4/*_%s.csv' \
						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder-w12/whisper-tiny.en-encoder-w12-798-lag10k-25-all-4/*_%s.csv' \
						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-798-lag10k-25-all-4/*_%s.csv' \
						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder-w12/whisper-tiny.en-encoder-w12-7170-lag10k-25-all-4/*_%s.csv' \
						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-7170-lag10k-25-all-4/*_%s.csv' \
					--labels whisper-de whisper-en whisper-de whisper-en whisper-de whisper-en whisper-de whisper-en \
					--model $$model \
					--keys $$key \
					--roi $$roi\
					--sig-elec-file tfs-sig-file-625-$$roi-$$model-$$key.csv tfs-sig-file-676-$$roi-$$model-$$key.csv tfs-sig-file-798-$$roi-$$model-$$key.csv tfs-sig-file-7170-$$roi-$$model-$$key.csv  \
					--fig-size $(FIG_SZ) \
					--lags-plot $(LAGS_PLT) \
					--lags-show $(LAGS_SHOW) \
					--x-vals-show $(X_VALS_SHOW) \
					$(LAG_TKS) \
					$(LAG_TK_LABLS) \
					$(PLT_PARAMS) \
					--outfile results/figures/tfs-$$model$(FN)-rois/tfs-$$roi-$$model$(FN)-$$key.pdf; \
			done; \
		done; \
	done;

# plot-whisper-rois:
# 	for model in $(MODELS); do \
# 		for roi in $(ROIS); do \
# 			for key in $(KEYS); do \
# 				python scripts/tfsplt_whisper.py \
# 					--sid 625 676 798 7170 \
# 					--formats \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-decoder/whisper-tiny.en-decoder-625-lag10k-25-all-3/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-625-lag10k-25-all-1/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-625-lag10k-25-all-2/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-625-lag10k-25-all-3/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-625-lag10k-25-all-4/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-decoder/whisper-tiny.en-decoder-676-lag10k-25-all-3/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-676-lag10k-25-all-1/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-676-lag10k-25-all-2/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-676-lag10k-25-all-3/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-676-lag10k-25-all-4/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-decoder/whisper-tiny.en-decoder-798-lag10k-25-all-3/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-798-lag10k-25-all-1/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-798-lag10k-25-all-2/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-798-lag10k-25-all-3/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-798-lag10k-25-all-4/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-decoder/whisper-tiny.en-decoder-7170-lag10k-25-all-3/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-7170-lag10k-25-all-1/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-7170-lag10k-25-all-2/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-7170-lag10k-25-all-3/*_%s.csv' \
# 						'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-7170-lag10k-25-all-4/*_%s.csv' \
# 					--labels whisper-de-3 whisper-en-1 whisper-en-2 whisper-en-3 whisper-en-4 whisper-de-3 whisper-en-1 whisper-en-2 whisper-en-3 whisper-en-4 whisper-de-3 whisper-en-1 whisper-en-2 whisper-en-3 whisper-en-4 whisper-de-3 whisper-en-1 whisper-en-2 whisper-en-3 whisper-en-4 \
# 					--model $$model \
# 					--keys $$key \
# 					--roi $$roi\
# 					--sig-elec-file tfs-sig-file-625-$$roi-$$model-$$key.csv tfs-sig-file-676-$$roi-$$model-$$key.csv tfs-sig-file-798-$$roi-$$model-$$key.csv tfs-sig-file-7170-$$roi-$$model-$$key.csv  \
# 					--fig-size $(FIG_SZ) \
# 					--lags-plot $(LAGS_PLT) \
# 					--lags-show $(LAGS_SHOW) \
# 					--x-vals-show $(X_VALS_SHOW) \
# 					$(LAG_TKS) \
# 					$(LAG_TK_LABLS) \
# 					$(PLT_PARAMS) \
# 					--outfile results/figures/tfs-$$model-all-layers-rois/tfs-$$roi-$$model-all-layers-$$key.pdf; \
# 			done; \
# 		done; \
# 	done;

plot-whisper:
	python scripts/tfsplt_whisper.py \
		--sid 625 676 798 7170 \
		--formats \
			'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-decoder/whisper-tiny.en-decoder-625-lag10k-25-all-3/*_%s.csv' \
			'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-625-lag10k-25-all-4/*_%s.csv' \
			'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-decoder/whisper-tiny.en-decoder-676-lag10k-25-all-3/*_%s.csv' \
			'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-676-lag10k-25-all-4/*_%s.csv' \
			'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-decoder/whisper-tiny.en-decoder-798-lag10k-25-all-3/*_%s.csv' \
			'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-798-lag10k-25-all-4/*_%s.csv' \
			'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-decoder/whisper-tiny.en-decoder-7170-lag10k-25-all-3/*_%s.csv' \
			'/scratch/gpfs/ln1144/247-encoding/results/tfs-whisper/whisper-tiny.en-encoder/whisper-tiny.en-encoder-7170-lag10k-25-all-4/*_%s.csv' \
		--labels whisper-de-l3 whisper-en-l4 whisper-de-l3 whisper-en-l4 whisper-de-l3 whisper-en-l4 whisper-de-l3 whisper-en-l4 \
		--keys $(KEY) \
		--roi $(ROI)\
		$(SIG_FN) \
		--fig-size $(FIG_SZ) \
		--lags-plot $(LAGS_PLT) \
		--lags-show $(LAGS_SHOW) \
		--x-vals-show $(X_VALS_SHOW) \
		$(LAG_TKS) \
		$(LAG_TK_LABLS) \
		$(PLT_PARAMS) \
		--outfile results/figures/tfs-all-rois/tfs-Precentral-all-comp.pdf

plot-new:
	rm -f results/figures/*
	python scripts/tfsplt_new.py \
		--sid 625 676 798 7170 \
		--formats \
			'/scratch/gpfs/kw1166/247-encoding/results/tfs/20230210-whisper-encoder-onset/kw-tfs-full-en-onset-625-whisper-tiny.en-l4-wn1-5/*/*_%s.csv' \
			'/scratch/gpfs/kw1166/247-encoding/results/tfs/20230212-whisper-decoder/kw-tfs-full-de-625-whisper-tiny.en-l3/*/*_%s.csv' \
			'/scratch/gpfs/kw1166/247-encoding/results/tfs/20230210-whisper-encoder-onset/kw-tfs-full-en-onset-676-whisper-tiny.en-l4-wn1-5/*/*_%s.csv' \
			'/scratch/gpfs/kw1166/247-encoding/results/tfs/20230212-whisper-decoder/kw-tfs-full-de-676-whisper-tiny.en-l3/*/*_%s.csv' \
			'/scratch/gpfs/kw1166/247-encoding/results/tfs/20230210-whisper-encoder-onset/kw-tfs-full-en-onset-798-whisper-tiny.en-l4-wn1-5/*/*_%s.csv' \
			'/scratch/gpfs/kw1166/247-encoding/results/tfs/20230212-whisper-decoder/kw-tfs-full-de-798-whisper-tiny.en-l3/*/*_%s.csv' \
			'/scratch/gpfs/kw1166/247-encoding/results/tfs/20230210-whisper-encoder-onset/kw-tfs-full-en-onset-7170-whisper-tiny.en-l4-wn1-5/*/*_%s.csv' \
			'/scratch/gpfs/kw1166/247-encoding/results/tfs/20230212-whisper-decoder/kw-tfs-full-de-7170-whisper-tiny.en-l3/*/*_%s.csv' \
		--labels whisper-en-l4 whisper-de-l3 whisper-en-l4 whisper-de-l3 whisper-en-l4 whisper-de-l3 whisper-en-l4 whisper-de-l3 \
		--keys comp prod\
		$(SIG_FN) \
		--fig-size $(FIG_SZ) \
		--lags-plot $(LAGS_PLT) \
		--lags-show $(LAGS_SHOW) \
		--x-vals-show $(X_VALS_SHOW) \
		$(LAG_TKS) \
		$(LAG_TK_LABLS) \
		$(PLT_PARAMS) \
		--outfile results/figures/ROI_encoding/test.pdf
	rsync -av results/figures/ ~/tigress/247-encoding-results


plot-twosplit:
	rm -f results/figures/*
	python scripts/tfsplt_newnew.py \
		--sid 247 \
		--formats \
			'results/uriplot2/data/247-glove-parstriangularis-all/*_%s.csv' \
			'results/uriplot2/data/247-glove-parstriangularis-all/*_%s.csv' \
			'results/uriplot2/data/247-gpt2n-parstriangularis-all/*_%s.csv' \
			'results/uriplot2/data/247-gpt2n-parstriangularis-all/*_%s.csv' \
			'results/uriplot2/data/247-gpt2n-1-parstriangularis-all/*_%s.csv' \
			'results/uriplot2/data/247-gpt2n-1-parstriangularis-all/*_%s.csv' \
			'results/uriplot2/data/247-glove-parsopercularis-all/*_%s.csv' \
			'results/uriplot2/data/247-glove-parsopercularis-all/*_%s.csv' \
			'results/uriplot2/data/247-gpt2n-parsopercularis-all/*_%s.csv' \
			'results/uriplot2/data/247-gpt2n-parsopercularis-all/*_%s.csv' \
			'results/uriplot2/data/247-gpt2n-1-parsopercularis-all/*_%s.csv' \
			'results/uriplot2/data/247-gpt2n-1-parsopercularis-all/*_%s.csv' \
			'results/uriplot2/data/247-glove-parsorbitalis-all/*_%s.csv' \
			'results/uriplot2/data/247-glove-parsorbitalis-all/*_%s.csv' \
			'results/uriplot2/data/247-gpt2n-parsorbitalis-all/*_%s.csv' \
			'results/uriplot2/data/247-gpt2n-parsorbitalis-all/*_%s.csv' \
			'results/uriplot2/data/247-gpt2n-1-parsorbitalis-all/*_%s.csv' \
			'results/uriplot2/data/247-gpt2n-1-parsorbitalis-all/*_%s.csv' \
		--keys \
			'glove comp parstriangularis' \
			'glove prod parstriangularis' \
			'gpt2-n comp parstriangularis' \
			'gpt2-n prod parstriangularis' \
			'gpt2-n-1 comp parstriangularis' \
			'gpt2-n-1 prod parstriangularis' \
			'glove comp parsopercularis' \
			'glove prod parsopercularis' \
			'gpt2-n comp parsopercularis' \
			'gpt2-n prod parsopercularis' \
			'gpt2-n-1 comp parsopercularis' \
			'gpt2-n-1 prod parsopercularis' \
			'glove comp parsorbitalis' \
			'glove prod parsorbitalis' \
			'gpt2-n comp parsorbitalis' \
			'gpt2-n prod parsorbitalis' \
			'gpt2-n-1 comp parsorbitalis' \
			'gpt2-n-1 prod parsorbitalis' \
		$(SIG_FN) \
		--fig-size $(FIG_SZ) \
		--lags-plot $(LAGS_PLT) \
		--lags-show $(LAGS_SHOW) \
		--x-vals-show $(X_VALS_SHOW) \
		$(LAG_TKS) \
		$(LG_TK_LABLS) \
		--lc-by 0 \
		--ls-by 1 \
		--split-hor 1 \
		--split-ver 2 \
		--outfile results/figures/tfs-brain-regions-allwords-allelecs.pdf
	rsync -av results/figures/ ~/tigress/247-encoding-results/13uriplots2


SIG_ELECS := --sig-elecs
# DIFF_AREA := --diff-area
# HAS_CTX := --has-ctx

CONDS := all correct incorrect all-flip
CONDS := all correct incorrect
CONDS := all shift-emb

plot-layers:
	rm -f results/figures/*
	python scripts/tfsplt_layer.py \
		--sid 625 \
		--layer-num 49 \
		--top-dir results/tfs/gpt2-layers-625 \
		--modes comp prod \
		--conditions $(CONDS) \
		$(HAS_CTX) \
		$(SIG_ELECS) \
		$(DIFF_AREA) \
		--outfile results/figures/test.pdf


# -----------------------------------------------------------------------------
# Miscellaneous
# -----------------------------------------------------------------------------

SP := 1

sig-test:
	rm -f results/figures/*
	python scripts/tfsmis_sig_test.py \
		--sid 676 \
		--formats \
			'results/tfs/kw-tfs-full-676-gpt2-xl-bert-lag10k-25-all/*/*_%s.csv' \
		--labels gpt-n-bert \
		--keys prod comp \
		--values $(LAGS) \
		$(SIG_FN) \
		--sig-percents $(SP)


# make sure the lags and the formats are in the same order
LAGS1 := {-10000..10000..25}
LAGS2 := -60000 -50000 -40000 -30000 -20000 20000 30000 40000 50000 60000
LAGS3 := -150000 -120000 -90000 90000 120000 150000
LAGS4 := -300000 -250000 -200000 200000 250000 300000
# LAGS_FINAL := -300000 -60000 -30000 {-10000..10000..25} 30000 60000 300000 # final
LAGS_FINAL := -99999999 # select all the lags that are concatenated (quardra)


concat-lags:
	python scripts/tfsenc_concat.py \
		--formats \
			'results/tfs/kw-tfs-full-7170-gpt2-xl-lag10k-25-all/*/' \
			'results/tfs/kw-tfs-full-7170-gpt2-xl-lag60k-10k-all/*/' \
			'results/tfs/kw-tfs-full-7170-gpt2-xl-lag150k-30k-all/*/' \
			'results/tfs/kw-tfs-full-7170-gpt2-xl-lag300k-50k-all/*/' \
		--lags \
			$(LAGS1) \
			$(LAGS2) \
			$(LAGS3) \
			$(LAGS4) \
		--lags-final $(LAGS_FINAL) \
		--output-dir results/tfs/plot-7170-gpt2-xl-quardra/kw-200ms-all-7170/


# plot-autocor:
# 	$(CMD) scripts/test.py





