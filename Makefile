
PDIR := $(shell dirname `pwd`)
USR := $(shell whoami | head -c 2)

# Sync electrode files (from /projects/HASSON/247/plotting)
# Link data files (from pickling/encoding/decoding results to data/)
# Create results folder for figures
link-data:
	mkdir -p data
	mkdir -p data/plotting
	rsync -rav /projects/HASSON/247/plotting/* data/plotting/
	mkdir -p data/pickling
	ln -fs $(PDIR)/247-pickling/results/* data/pickling/
	mkdir -p data/encoding
	ln -fs $(PDIR)/247-encoding/results/* data/encoding/
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