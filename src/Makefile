all:
	g++ -I ../eigen/ -O3  -march=native -O4 -Wall -fopenmp -std=c++0x  reps.cc -o reps

pre:
	./reps --dim=300 --epohs=20 --model=../work/joint_model --alpha=0.01 --lmda=10000 --edges=../work/edges.bench --pretrain=../work/glove300.txt --pairs=../work/synonyms.bench.comb

clean:
	rm ../work/joint_model ../work/res.csv

