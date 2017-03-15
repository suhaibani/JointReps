"""
Combine the lexicon and the co-occurrence file.
"""

import numpy
import sys
def process(lexicon_fname, cooc_fname, comb_fname):
    # load all the words in the lexicon.
    G = {}
    with open(lexicon_fname) as lexicon_file:
        for line in lexicon_file:
            p = line.strip().split('\t')
            #assert(len(p) == 2)
            first = p[0]
            second = p[1]
            G.setdefault(first, {'lexicon':set(), 'corp':{}})
            G[first]['lexicon'].add(second)
    count = 0
    for word in G:
        count += len(G[word]['lexicon'])
    print "Total no. of vertices in the lexicon graph =", len(G)
    print "Average no. of neigbours per vertex (degree) =", float(count) / float(len(G))

    # add the co-occurrences.
    freq = {}
    N = 0
    lineCounter=0
    
    with open(cooc_fname) as cooc_file:
        for line in cooc_file:
            p = line.strip  ().split('\t')
            assert(len(p) == 3)
            first = p[0]
            second = p[1]
            val = float(p[2])
            lineCounter+=1
            if lineCounter%50000==0:
                print lineCounter
            if first in G:
                G[first]['corp'][second] = G[first]['corp'].get(second, 0.0) + val
            if second in G:
                G[second]['corp'][first] = G[second]['corp'].get(first, 0.0) + val
            freq[first] = freq.get(first, 0) + val
            freq[second] = freq.get(second, 0) + val
            N += 2 * val

    # convert the co-occurrence into PPMI values.
    conv_PPMI(G, freq, N)

    count = 0
    for word in G:
        count += len(G[word]['corp'])
    print "Total no. of vertices in the cooc graph =", len(G)
    print "Average no. of neigbours per vertex (degree) =", float(count) / float(len(G))
    count = 0
    topK = 5
    for word in G:
        L = G[word]['corp'].items()
        L.sort(lambda x, y: -1 if x[1] > y[1] else 1)
        G[word]['corp'] = {}
        for (key, val) in L[:topK]:
            G[word]['corp'][key] = val
        count += len(G[word]['corp'].keys()) + len(G[word]['lexicon'])
    print "Total no. of vertices in the entire graph =", len(G)
    print "Average no. of neigbours per vertex (degree) =", float(count) / float(len(G))

    # normalize and write the graph.
    with open(comb_fname, 'w') as comb_file:
        for target in G:
            for context in G[target]['lexicon']:
                comb_file.write("%s\t%s\t1.0\n" % (target, context))
            total = sum(G[target]['corp'].values())
            for (context, val) in G[target]['corp'].iteritems():
                comb_file.write("%s\t%s\t%f\n" % (target, context, float(val) / float(total)))
    pass

def conv_PPMI(G, freq, N):
    for target in G:
        h = {}
        for context in G[target]['corp']:
            val = numpy.log((G[target]['corp'][context] * N) / (freq[target] * freq[context]))
            if val > 0:
                h[context] = val
        G[target]['corp'] = h
    pass




if __name__ == "__main__":
    
    process(sys.argv[1],sys.argv[2],sys.argv[3])
    pass
