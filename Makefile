MOSTGIM = linear

all: run

run:
	python3 prepare.py
	python3 mosgim_$(MOSTGIM).py
	python3 createLCP.py
	python3 plotN.py
