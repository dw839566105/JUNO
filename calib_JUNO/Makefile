.PHONY: all

radius:=$(shell seq -17000 1000 17000)
radius:=$(shell seq 1000 1000 17000)

save_pe:=JUNO_test

src_pe:=$(radius:%=$(save_pe)/file_%.h5)

order:=30

all:$(src_pe)

$(save_pe)/file_%.h5:
	mkdir -p $(save_pe)
	python3 PE_calib/calib_new.py $* $@ $(order) > logs/PE$*.log 2>&1
.SECONDARY:
