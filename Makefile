# TODO


# commands for plotting (maybe don't need any because of jupyter notebooks?)
# clean up plotting scripts (find the paralleled ones)


PDIR := $(shell dirname `pwd`)
USR := $(shell whoami | head -c 2)

# Link electrode files (from /projects/HASSON/247/plotting)
# Link data files (from pickling/encoding/decoding results to data/)
# Create results folder for figures
set-up:
	mkdir -p data
	mkdir -p data/plotting
	rsync -rav /projects/HASSON/247/plotting/* data/plotting/
	mkdir -p data/pickling
	ln -fs $(PDIR)/247-pickling/results/* data/pickling/
	mkdir -p data/encoding
	ln -fs $(PDIR)/247-encoding/results/* data/encoding/
	mkdir -p results
	mkdir -p results/figures


sync-data:
	rsync -rav /projects/HASSON/247/plotting/* data/plotting/



LAYER_IDX := $(shell seq 10 24)
LAYER_IDX := $(shell seq 0 22)
LAYER_IDX := 23

CMD := echo
CMD := sbatch submit1.sh
CMD := python


emb-tsne:
	for layer in $(LAYER_IDX); do \
		$(CMD) scripts/tfsemb_emb-tsne.py \
			--layer $$layer; \
	done;
	
# $(CMD) scripts/tfsemb_emb-tsne.py \ 
# 	--layer $(LAYER_IDX);
