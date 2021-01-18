JUNO:=$(wildcard /junofs/users/junoprotondecay/xubd/harvest/det/e-/0_0_1/2/?000/0[0-2].h5)
dst:=$(JUNO:/junofs/users/junoprotondecay/xubd/harvest/det/e-/0_0_1/2/%=result/%)

all:$(dst)

result/%.h5: /junofs/users/junoprotondecay/xubd/harvest/det/e-/0_0_1/2/%.h5
	mkdir -p $(shell dirname $@)
	python3 Recon_test.py $^ $@ > $@.log 2>&1

.DELETE_ON_ERROR:

.SECONDARY:
