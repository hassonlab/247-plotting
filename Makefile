
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
	# rsync -rav /projects/HASSON/247/plotting/* data/plotting/
	ln -fs $(PDIR)/../../247/247-pickling/results data/pickling
	ln -fs $(PDIR)/../../247/247-encoding-dev/results data/encoding
	mkdir -p results
	mkdir -p results/figures

# NOT tested, I have write prvlgs in the plotting folder
# link-data-local-dirs:
#     mkdir -p data
#     mkdir -p data/plotting
#     # rsync -rav /projects/HASSON/247/plotting/* data/plotting/
#     find $(PDIR)/../../247/247-pickling/results -type d -exec mkdir -p data/pickling/{} \;
#     find $(PDIR)/../../247/247-pickling/results -type f -exec ln -fs $(PDIR)/../../247/247-pickling/results/{} data/pickling/{} \;
#     find $(PDIR)/../../247/247-encoding-dev/results -type d -exec mkdir -p data/encoding/{} \;
#     find $(PDIR)/../../247/247-encoding-dev/results -type f -exec ln -fs $(PDIR)/../../247/247-encoding-dev/results/{} data/encoding/{} \;
#     mkdir -p results
#     mkdir -p results/figures

# resync data from projects
sync-data:
	rsync -rav /projects/HASSON/247/plotting/* data/plotting/



######################################################################################
####################################  Embedding  #####################################
######################################################################################


# layer index (use -1 for set layers)
LAYER_IDX := -1
LAYER_IDX := $(shell seq 17 32)

# whether to aggregate and average datum (comment out to not run this step)
AGGR := --aggregate
AGGR :=

# whether to perform tsne (comment out to not run this step)
TSNE :=
TSNE := --tsne
PCA := --pca
PCA :=

# whether to perform classification (comment out to not run this step)
CLASS :=
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
		$(PCA) \
		$(CLASS) \
		--aggr-type $(AGGR_TYPE) \
		--savedir results/20240212-podcast-pkl-from-daria \
		--xcol de_emb \
		--ycol manner_artic \
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


emb-class-mia-layers:
	for layer in $(LAYER_IDX); do \
		$(CMD) scripts/tfsemb_class-mia.py \
			$(AGGR) \
			--savedir results/20240626-mia-classify \
			--emb opt-2.7b \
			--context 2048 \
			--layer $$layer; \
	done;



######################################################################################
#####################################  Encoding  #####################################
######################################################################################

# make sure the lags and the formats are in the same order
LAGS2 := -60000 -50000 -40000 -30000 -20000 20000 30000 40000 50000 60000
LAGS3 := -150000 -120000 -90000 90000 120000 150000
LAGS4 := -300000 -250000 -200000 200000 250000 300000
LAGS_FINAL := -300000 -60000 -30000 {-10000..10000..25} 30000 60000 300000 # final
LAGS_FINAL := {-5000..5000..25}
LAGS1 := {-10000..10000..25}
# LAGS_FINAL := -99999999 # select all the lags that are concatenated (quardra)

SID:= 625
SID:= 676
SID:= 798
SID:= 7170
concat-lags:
	python scripts/tfsenc_concat.py \
		--formats \
			'data/encoding/tfs/paper-prob-improb/20230809-gpt2-preds/kw-tfs-full-$(SID)-gpt2-xl-glove50-lag10k-25-aligned-improb/kw-200ms-all-$(SID)/' \
		--lags \
			$(LAGS1) \
		--lags-final $(LAGS_FINAL) \
		--output-dir data/encoding/tfs/20240522-glove-5k/kw-tfs-full-$(SID)-gpt2-xl-glove50-lag10k-25-aligned-improb/kw-200ms-all-$(SID)/



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

LAGS_PLT := {-1000..1000..25} # lag1k-25
LAGS_PLT := {1..1..1} # lag2k-25 for pred-lag
LAGS_PLT := {-5000..5000..25} # lag5k-25
LAGS_PLT := {-10000..10000..50} # lag10k-25
LAGS_PLT := {-2000..2000..25} # lag2k-25
LAGS_PLT := {-60000..60000..50} # lag60k-50
LAGS_PLT := {-30000..30000..50} # lag30k-50
LAGS_PLT := {-120000..120000..50} # 


# zoomed-in version (from -2s to 2s)
LAGS_SHOW := {-2000..2000..25}
LAGS_SHOW := {-5000..5000..50}
X_VALS_SHOW := {-2000..2000..25}
X_VALS_SHOW := {-5000..5000..50}
LAG_TKS := 
LAG_TK_LABLS :=

# Plotting for vanilla encoding (no concatenated lags)
# LAGS_SHOW := {-2000..2000..50}
# LAGS_SHOW := $(LAGS_PLT)
# LAGS_SHOW := {-500..0..25}
# X_VALS_SHOW := $(LAGS_SHOW)
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
SIG_FN_DIR := 'data/plotting/sig-elecs/20230510-tfs-sig-file'
SIG_FN_DIR := 'data/plotting/sig-elecs/20230405-ccn'
SIG_FN_DIR := 'data/plotting/sig-elecs/20231201-eric-plots'
SIG_FN_DIR := 'data/plotting/sig-elecs'
SIG_FN_DIR := 'data/plotting/sig-elecs/20240510-tfs-sig-file'
SIG_FN_DIR := 'data/plotting/sig-elecs/20230723-tfs-sig-file'
SIG_FN_DIR :=
SIG_FN_DIR := 'data/plotting/sig-elecs/20230413-whisper-paper'
SIG_FN_DIR := 'data/plotting/sig-elecs/20230723-tfs-sig-file'
SIG_FN_DIR := 'data/plotting/sig-elecs/20240303-tfs-sig-file-corr-01'
SIG_FN_DIR := 'data/plotting/sig-elecs/20250415-tfs-sig-file-corr-01'


# Significant electrode files
SIG_FN := --sig-elec-file tfs-sig-file-%s-whisper-ende-outer-comp.csv tfs-sig-file-%s-whisper-ende-outer-prod.csv
SIG_FN := --sig-elec-file %s-sig-elecs_comp.csv %s-sig-elecs_prod.csv
SIG_FN := --sig-elec-file %s-ifg-elecs-comp.csv %s-ifg-elecs-comp.csv
SIG_FN := --sig-elec-file tfs-sig-file-%s-whisper-ac-last-0.01-comp.csv tfs-sig-file-%s-whisper-ac-last-0.01-prod.csv
SIG_FN := --sig-elec-file tfs-sig-file-%s-whisper-de-best-0.01-comp.csv tfs-sig-file-%s-whisper-de-best-0.01-prod.csv
SIG_FN := --sig-elec-file podcast_160.csv
SIG_FN := --sig-elec-file tfs-sig-file-%s-whisper-varpar-acac-outer-comp.csv tfs-sig-file-%s-whisper-varpar-acac-outer-prod.csv
SIG_FN := --sig-elec-file %s-llama3-sig-comp.csv %s-llama3-sig-prod.csv
SIG_FN := --sig-elec-file tfs-sig-file-%s-whisper-varpar-ende-outer-comp.csv tfs-sig-file-%s-whisper-varpar-ende-outer-prod.csv
SIG_FN := --sig-elec-file tfs-sig-file-%s-whisper-en-last-0.01-comp.csv tfs-sig-file-%s-whisper-en-last-0.01-prod.csv
SIG_FN := --sig-elec-file tfs-sig-file-glove-%s-comp.csv tfs-sig-file-glove-%s-prod.csv
SIG_FN := --sig-elec-file %s-banded-mist-gpt2-sig-comp.csv %s-banded-mist-gpt2-sig-prod.csv
SIG_FN := --sig-elec-file %s-banded-mist-gpt2-sig-comp.csv %s-banded-mist-gpt2-sig-prod.csv
SIG_FN := 

		# --formats \
		# 	'/scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-798-gpt2-xl-bandedRidge-lag10k-50-all-static/*/*_%s_banded_joint.csv' \
		# 	'/scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-798-gpt2-xl-bandedRidge-lag10k-50-all-static/*/*_%s_banded_sentence.csv' \
		# 	'/scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-798-gpt2-xl-bandedRidge-lag10k-50-all-static/*/*_%s_banded_word.csv' \
		# --labels joint sentence word\

		# --formats \
		# 	'/scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-798-gpt2-xl-Ridge-lag30k-25-all-mistral-sent-only/*/*_%s.csv' \
		# --labels sentence-only \

		# --formats \
		# 	'/scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-798-gpt2-xl-ridge-lag2k-25-all/*/*_%s.csv' \
		# --labels glove \

		# 2 word models:
		# ij-tfs-798-gpt2-xl-bandedRidge-lag2k-50-all-static-n-cont-n-1-2words
		# ij-tfs-798-gpt2-xl-bandedRidge-lag2k-50-all-static-mistral-sent-2words

		# ij-tfs-%s-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past-reph-translate_pca300
		# ij-tfs-%s-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past-reph-translate-control_pca300

# 		ij-tfs-676-gpt2-xl-bandedRidge-lag60-50-all-static_future_past-reph-translate-control_pca300_drop-short_mistral
# 		ij-tfs-676-gpt2-xl-bandedRidge-lag60-50-all-static_stack_merge_5_pca300_drop-short
#     ij-tfs-798-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past-no-reph_2words_punct-sents_drop-short
#     ij-tfs-798-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past-reph_2words_punct-sents_drop-short

# 		--labels joint word_n1_4 word\

# 		--formats \
# 			'/scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-798-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past-reph_2words_punct-sents_drop-short/*/*_%s_banded_joint.csv' \
# 			'/scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-798-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past-reph_2words_punct-sents_drop-short/*/*_%s_banded_sentence.csv' \
# 			'/scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-798-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past-reph_2words_punct-sents_drop-short/*/*_%s_banded_sentence2.csv' \
# 			'/scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-798-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past-reph_2words_punct-sents_drop-short/*/*_%s_banded_word.csv' \
# 			'/scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-798-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past-no-reph_2words_punct-sents_drop-short/*/*_%s_banded_joint.csv' \
# 			'/scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-798-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past-no-reph_2words_punct-sents_drop-short/*/*_%s_banded_sentence.csv' \
# 			'/scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-798-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past-no-reph_2words_punct-sents_drop-short/*/*_%s_banded_sentence2.csv' \
# 			'/scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-798-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past-no-reph_2words_punct-sents_drop-short/*/*_%s_banded_word.csv' \
# 		--labels reph_joint reph_fut reph_past reph_word joint fut past word\

# ij-tfs-676-gpt2-xl-Ridge-lag30k-50-all-static_future_only_drop-short_mistral_no-word
# ij-tfs-676-gpt2-xl-Ridge-lag30k-50-all-static_past_only_drop-short_mistral_no-word
# ij-tfs-676-gpt2-xl-Ridge-lag30k-50-all-static_static_word_only_drop-short

# ij-tfs-${SID}-gpt2-xl-bandedRidge-lag60-50-all-static_future_past-reph-translate-control_pca300_drop-short_mistral
#  ij-tfs-${SID}-gpt2-xl-bandedRidge-lag30k-50-all-static_future_past-reph-translate-control_30s_drop-short_mistral

SID:= 625

plot-encoding:
	# rm -f results/figures/*
	python scripts/tfsplt_encoding.py \
		--sid $(SID) \
		--formats \
			'/scratch/gpfs/HASSON/gidon/247-encoding-dev/results/tfs/gp-tfs-625-gpt2-xl-bandedRidge-lag60-50-all-static_future_past-reph-translate-control_pca300_drop-short_mistral_untrained_original_sentence/gp-200ms-625/*_%s_banded_joint.csv' \
			'/scratch/gpfs/HASSON/gidon/247-encoding-dev/results/tfs/gp-tfs-625-gpt2-xl-bandedRidge-lag60-50-all-static_future_past-reph-translate-control_pca300_drop-short_mistral_untrained_original_sentence/gp-200ms-625/*_%s_banded_sentence.csv' \
			'/scratch/gpfs/HASSON/gidon/247-encoding-dev/results/tfs/gp-tfs-625-gpt2-xl-bandedRidge-lag60-50-all-static_future_past-reph-translate-control_pca300_drop-short_mistral_untrained_original_sentence/gp-200ms-625/*_%s_banded_sentence2.csv' \
			'/scratch/gpfs/HASSON/gidon/247-encoding-dev/results/tfs/gp-tfs-625-gpt2-xl-bandedRidge-lag60-50-all-static_future_past-reph-translate-control_pca300_drop-short_mistral_untrained_original_sentence/gp-200ms-625/*_%s_banded_word.csv' \
		--labels joint future past word\
		--keys comp prod \
		--sig-elec-file-dir $(SIG_FN_DIR)\
		$(SIG_FN) \
		--fig-size $(FIG_SZ) \
		--lags-plot {-60000..60000..50} \
		--lags-show {-2000..2000..50} \
		--x-vals-show {-2000..2000..50} \
		$(LAG_TKS) \
		$(LAG_TK_LABLS) \
		$(PLT_PARAMS) \
		--y-vals-limit $(Y_LIMIT) \
		--outfile /scratch/gpfs/HASSON/gidon/247-plotting/results/figures/$(SID)_bandedRidge_future_past_dropshort_5s.pdf

plot-encoding-multiple-models:
	# rm -f results/figures/*
	python scripts/tfsplt_encoding.py \
		--sid 798 \
		--formats \
			'/scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-798-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past_pca300_drop-short_mistral_low-info/*/*_%s_banded_joint.csv' \
			'/scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-798-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past_pca300_drop-short_mistral_high-info/*/*_%s_banded_joint.csv' \
		--labels low-inf high-inf\
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
		--outfile results/figures/798_bandedRidge_gpt2_stat_mistral_all_elecs_pca_drop-short_low_v_high-info.pdf


plot-encoding-layers:
	rm -f results/figures/*
	python scripts/tfsplt_encoding-layers.py \
		--sid 777 \
		--formats \
			'data/encoding/podcast/20230424-bert/kw-podcast-full-%s-bert-large-uncased-wwm-lag5k-25-all-%s/*/*_%s_fold.csv' \
		--labels $(shell seq 0 24) \
		--colors coolwarm \
		--keys comp \
		--sig-elec-file-dir $(SIG_FN_DIR)\
		$(SIG_FN) \
		--fig-size $(FIG_SZ) \
		--lags-plot $(LAGS_PLT) \
		--lags-show $(LAGS_SHOW) \
		--x-vals-show $(X_VALS_SHOW) \
		$(LAG_TKS) \
		$(LAG_TK_LABLS) \
		--y-vals-limit $(Y_LIMIT) \
		--x-label layer \
		--outfile results/figures/eric-plots.pdf
	rsync -av results/figures/ ~/tigress/247-encoding-results/


plot-brainmap:
	python scripts/tfsplt_brainmap.py \
		--sid 625 676 7170 798 \
		--formats \
			'data/encoding/tfs/20250113-sts/kw-tfs-%s-seamless-m4t-v2-large-tts-dec-short-ridge-lag2k-25-all-l13/*/*_%s.csv' \
		--effect max \
		--keys comp prod \
		--lags-plot $(LAGS_PLT) \
		--lags-show $(LAGS_SHOW) \
		--sig-elec-file-dir $(SIG_FN_DIR) \
		$(SIG_FN) \
		--final \
		--outfile fig_%s13.png
	rsync -av results/figures/ ~/tigress/247-encoding-results/


# LAGS_SHOW := {0..1000..25}
# LAGS_SHOW := {-1000..0..25}
# color direction - 2-->1 (2nd format > 1st == low; 1st format > 2nd == high)

# ij-tfs-%s-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past_pca300_drop-short_mistral_high-info
# ij-tfs-%s-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past_pca300_drop-short_mistral_low-info

# /scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-%s-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past_pca300_drop-short_mistral_high-info/*/*_%s_banded_joint.csv \
# /scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-%s-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past_pca300_drop-short_mistral_low-info/*/*_%s_banded_joint.csv \

#     ij-tfs-%s-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past-no-reph_2words_punct-sents_drop-short
#     ij-tfs-%s-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past-reph_2words_punct-sents_drop-short
#     ij-tfs-%s-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past-no-reph_last2words_punct-sents_drop-short
#     ij-tfs-%s-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past-reph_last2words_punct-sents_drop-short

# grad direction: pos: 2>1, neg: 1>2

plot-brainmap-2d:
	python scripts/tfsplt_brainmap_2d.py \
		--sid 625 676 798 \
		--formats \
			/scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-%s-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past-reph_2words_punct-sents_drop-short/*/*_%s_banded_joint.csv \
			/scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-%s-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past-no-reph_2words_punct-sents_drop-short/*/*_%s_banded_joint.csv \
		--effect varpar \
		--keys comp prod \
		--cmap PU_RdBu_covar \
		--lags-plot $(LAGS_PLT) \
		--lags-show $(LAGS_SHOW) \
		--sig-elec-file-dir $(SIG_FN_DIR) \
		$(SIG_FN) \
		--final \
		--outfile results/figures/banded_high_low_info_2d_0_500_01_elecs_%s_sig01_0_500.jpeg
	# rsync -av results/figures/ ~/tigress/247-encoding-results/


plot-brainmap-subjects:
	python scripts/tfsplt_brainmap_cat.py \
		--sid 625 676 7170 798 \
		--keys comp prod \
		--sig-elec-file-dir $(SIG_FN_DIR) \
		$(SIG_FN) \
		--outfile fig_%s.png
	rsync -av results/figures/ ~/tigress/247-encoding-results/

# --formats \
# 			/scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-%s-gpt2-xl-bandedRidge-lag2k-25-all-static/*/*_%s_banded_sentence.csv \
# 			/scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-%s-gpt2-xl-bandedRidge-lag2k-25-all-static/*/*_%s_banded_word.csv \

# effect can be gradient or color, it's ignored if plotting one map
# need to change vmin and vmax in code 
# if doing gradient, directionality is 2-1 (pos is 2>1)
# /scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-%s-gpt2-xl-bandedRidge-lag2k-50-all-static-mistral-sent-2words/*/*_%s_banded_joint.csv \
# /scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-%s-gpt2-xl-bandedRidge-lag2k-50-all-static-n-cont-n-1-2words/*/*_%s_banded_joint.csv \
#
# lags-plot is the datasize, lags-show is the range used in the plot
# grad direction: pos: 2>1, neg: 1>2

plot-glassbrain:
	python scripts/tfsplt_glassbrain.py \
		--sid 625 676 798 7170 \
		--formats \
			/scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-%s-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past-reph_last2words_punct-sents_drop-short/*/*_%s_banded_joint.csv \
			/scratch/gpfs/ij9216/projects/code/247/247-plotting/data/encoding/tfs/ij-tfs-%s-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past-no-reph_last2words_punct-sents_drop-short/*/*_%s_banded_joint.csv \
		--effect gradient \
		--keys comp prod \
		--cmap  PiYG \
		--lags-plot $(LAGS_PLT) \
		--lags-show $(LAGS_SHOW) \
		--sig-elec-file-dir $(SIG_FN_DIR) \
		$(SIG_FN) \
		--final \
		--outfile results/figures/banded_all_subs_01_elecs_reph_no-reph_last2w_%s_-1000_0_joint.jpeg


sig-elecs:
	python scripts/tfsplt_sigelecs.py \
		--sid 625 7170 798 \
		--formats \
			'data/encoding/tfs/20250216-sig-elecs/kw-tfs-%s-seamless-m4t-v2-large-tts-dec0-short-sig-elec-lag2k-25-all/*/*_%s.csv' \
			'data/encoding/tfs/20250213-sts-bridge/kw-tfs-%s-all3/*/*_%s.csv' \
		--effect pr \
		--keys comp prod \
		--cmap viridis \
		--lags-plot $(LAGS_PLT) \
		--lags-show $(LAGS_SHOW) \
		--sig-elec-file-dir $(SIG_FN_DIR) \
		$(SIG_FN) \
		--final \
		--outfile fig_%s.jpeg
	rsync -av results/figures/ ~/tigress/247-encoding-results/