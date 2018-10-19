all:
	g++ -I ../eigen/ -O3 -march=native -O4 -Wall -fopenmp -std=c++0x reps.cc -o reps

run:
	./reps --dim=300 --epohs=20 --model=../work/joint_model --alpha=0.01 --lmda=10000 --edges=../work/edges.bench --pretrain=../work/glove300.txt --pairs=../work/synonyms.bench.comb
