.PHONY: all

path:=/mnt/stage/douwei/Simulation/1t_root/2MeV_015/
radius:=$(shell seq -f '%+.2f' +0.60 -0.01 -0.60)
save:=coeff_pe_1t_2.0MeV
src:=$(radius:%=$(save)/file_%.h5)

all:$(src)

$(save)/file_%.h5:
	mkdir -p $(save)
	python3 PE_calib/main_calib.py $* $(path) $@
