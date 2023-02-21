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
	mkdir -p results
	mkdir -p data/
	ln -sf /projects/HASSON/247/plotting data/
	mkdir -p data/pickling
	ln -fs $(PDIR)/247-pickling/results/* data/pickling/
	mkdir -p data/encoding
	ln -fs $(PDIR)/247-encoding/results/* data/encoding/
