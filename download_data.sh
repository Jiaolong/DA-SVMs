#!/bin/csh
# Dowload Office and Caltech dataset
# Copyright (c) 2015-2016, Jiaolong Xu. jiaolong@cvc.uab.es

if (! -e data) then
	wget http://www-scf.usc.edu/%7Eboqinggo/domain_adaptation/GFK_v1.zip
	unzip GFK_v1.zip
	mv ToRelease_GFK/data/ .
	rm -rf ToRelease_GFK
	rm -f GFK_v1.zip
else
	echo "data already exits!"
	ls data
endif
