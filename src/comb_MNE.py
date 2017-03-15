"""
Combine the lexicon and the co-occurrence file.

"""

import numpy
import sys
def process(lexicon_fname, cooc_fname, comb_fname,connectedNodes,wn_terms):
    # load all the words in the lexicon.
    G = {}
    r=1
    with open(lexicon_fname) as lexicon_file:
        for line in lexicon_file:
            p = line.strip().split('\t')
            assert(len(p) == 2)
            first = p[0]
            second = p[1]
            #if first in connectedNodes and second in connectedNodes:
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
            p = line.strip().split('\t')
            assert(len(p) == 3)
            first = p[0]
            second = p[1]
            val = float(p[2])
            if first in connectedNodes and second in connectedNodes:
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
    count=0
    common_from_corpus=open('common.txt',"wb")
    gLength=len(G)
    print "Generating nearest neighbour candidates.."
    print "Total Tokens..",gLength
    for word in G:
        L = G[word]['corp'].items()
        L.sort(lambda x, y: -1 if x[1] > y[1] else 1)
        # get top k corpus words for each word
        # for each corpus word in topk
        for (key, val) in L[:topK]:
            # for each synonym of the word,get its linked corpus words
            for term in G[word]['lexicon']:
                if term in G:
                    synsets= G[term]['corp'].items()
                
                    synsets.sort(lambda x, y: -1 if x[1] > y[1] else 1)
                #take topk corpus words
                    temp2=synsets[:topK]
                    for (x,y) in temp2:
                        if key == x:
                            connectedNodes[key]+=1
        count+=1
        if count%500==0:
        	print "processed..",count,' out of ',gLength
    '''
    noise=open('noise_from_corpus','wb')
    
    for key in connectedNodes:
        if connectedNodes[key]==0:
            noise.write(key)
            noise.write('\n')
    noise.close()
    '''
    nearest_candidates={}
    common_from_corpus.close()
    for key in connectedNodes:
    	if connectedNodes[key]>=r:
    		nearest_candidates[key]=1
    print "Completed.."
    wn_terms.clear()
    connectedNodes.clear()
    print "nearest candidates..",len(nearest_candidates)
    print "Generating Graph based on R and nearest neighbours.."
    for word in G:
        L = G[word]['corp'].items()
        L.sort(lambda x, y: -1 if x[1] > y[1] else 1)
        good_candidates={}
        for (key,value) in L:
        	if key in nearest_candidates:
        		good_candidates[key]=value
        sorted_candidates=good_candidates.items()
        #good_candidates.clear()
        sorted_candidates.sort(lambda x, y: -1 if x[1] > y[1] else 1)

        G[word]['corp'] = {}
        for (key, val) in sorted_candidates[:topK]:
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

def newApproach(lexicon_file,newFile):
    # unique words extracted from synonyms file (wordnet relations)
    wn_terms={}
    with open (lexicon_file) as wordnet:
        for line in wordnet:
            w1=line.strip().split()[0]
            w2=line.strip().split()[1]
            if w1 not in wn_terms:
                wn_terms[w1]=1
            if w2 not in wn_terms:
                wn_terms[w2]=1
    node={}
    r=1
    print "Opening",newFile,"....."
    lineC=0
    with open(newFile) as words:
        for term in words:
            if lineC%100000==0:
                print "Processed...",lineC
                

            w1=term.strip().split('\t')[0]
            w2=term.strip().split('\t')[1]
            # inserting word in dictionary node_from_corp
            if  w1 not in node:# and w1 not in wn_terms:
                node[w1]=0
            if w2 not in node:# and w2 not in wn_terms:
                node[w2]=0

            #building connection , of either other the word from pair is synonym, increase the count fot the non synonym
            #if w1 not in wn_terms and w2 in wn_terms:
            if w2 in wn_terms:
                node[w1]+=1
            if w1 in wn_terms:
                node[w2]+=1
            lineC+=1
    connectedNodes={}
    accepted=0.0
    total=0.0
    for key in node:
        total+=1
        if node[key]>=r:
            accepted+=1
            #print key,node[key]
            connectedNodes[key]=0
       
    print ""
    print "Statistics.."
    print "Total accepted..",accepted
    print "Acceptance Rate..",float((accepted/total)*100.0),"%"
    print "Noise removed..",float(((total-accepted)/total)*100.0),"%"
    return (connectedNodes,wn_terms)


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
    lexicon_file=sys.argv[1]
    output=newApproach(lexicon_file,sys.argv[2])
    connectedNodes=output[0]
    wn_terms=output[1]
    process(sys.argv[1], sys.argv[2], sys.argv[3],connectedNodes,wn_terms)
    pass

