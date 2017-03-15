#include <iostream>
#include <vector>
#include <list>
#include <set>
#include <tuple>
#include <unordered_map>
#include <cstdio>
#include <utility>
#include <cstdlib>
#include <map>
#include <cmath>
#include <ctgmath>
#include <cstring>
#include <functional>
#include <cassert>
#include <algorithm>
#include <unordered_map>
#include <fstream>
#include <stdio.h>

#include <boost/algorithm/string.hpp>
#include <Eigen/Dense>
#include "parse_args.hh"
#include "omp.h"

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

using namespace std;
using namespace Eigen;

int D; // dimensionality

unordered_map<string, VectorXd> w; // target word vectors
unordered_map<string, VectorXd> c; // context word vectors
unordered_map<string, double> bw; // target word biases
unordered_map<string, double> bc; // context word biases

unordered_map<string, VectorXd> grad_w; // Squared gradient for AdaGrad
unordered_map<string, VectorXd> grad_c; 
unordered_map<string, double> grad_bw; 
unordered_map<string, double> grad_bc; 

struct edge {
  string w, c;
  double value;
};
vector<edge> edges;

set<string> words;
set<string> contexts;

unordered_map<string, double> pairs;


void read_edges(string fname, vector<edge> &train_data){
    ifstream train_file(fname.c_str());
    string first, second;
    double value;
    while (train_file >> first >> second >> value){
        edge e;
        e.w = first;
        e.c = second;
        e.value = value;
        //cout << e.w << "\t" << e.c << "\t" << e.value << endl;
        words.insert(first);
        contexts.insert(second);
        train_data.push_back(e);
    }
    train_file.close();
    // allocate memory for all vectors
}  

void load_word_vectors(string vects_fname){
    // Initialize word vectors using pre-trained word vectors
    fprintf(stderr, "%sDimensionality of the word vectors = %d%s\n", KRED, D, KNRM);
    fprintf(stderr, "%sReading pre-trained vectors from %s%s\n", KRED, vects_fname.c_str(), KNRM);
    unordered_map<string, VectorXd> x;
    set<string> prewords;

    FILE *fp= fopen(vects_fname.c_str(), "r");
    int count = 0;
    for (char buf[262144]; fgets(buf, sizeof(buf), fp); ) {
        char curword[1024];
        sscanf(buf, "%s %[^\n]", curword, buf);
        string wstr(curword);
        if ((words.find(wstr) != words.end()) || (contexts.find(wstr) != contexts.end())){
            x[wstr] = VectorXd::Zero(D);
            count = 0;
            while (1) {
                double fval;
                bool end = (sscanf(buf, "%lf %[^\n]", &fval, buf) != 2);
                x[wstr][count] = fval;
                count += 1;
            if (end) break;
            }
            prewords.insert(wstr);
            assert(count == D);
        }
    }
    // Copy the vectors from x to w and c.
    for (auto y = x.begin(); y != x.end(); ++y){
        if (words.find(y->first) != words.end())
            w[y->first] = y->second;
        if (contexts.find(y->first) != contexts.end())
            c[y->first] = y->second;
    }
    fclose(fp);
}

void read_pairs(string pairs_fname){
    fprintf(stderr, "%sReading relational pairs from = %s%s\n", KRED, pairs_fname.c_str(), KNRM);
    ifstream pairs_file(pairs_fname.c_str());
    string first_word, second_word;
    double value;
    
    while (pairs_file >> first_word >> second_word >> value){
        //cout << first_word << "\t" << second_word << "\t" << value << "\t" << (2 * value) << endl;
        pairs[first_word + "<+>" + second_word] = value;
    }
    
    /*
    while (pairs_file >> first_word >> second_word){
        //cout << first_word << "\t" << second_word << "\t" << value << endl;
        pairs[first_word + "<+>" + second_word] = 1.0;
    }
    */

    pairs_file.close();
}

void centralize(unordered_map<string, VectorXd> &x){
    VectorXd mean = VectorXd::Zero(D);
    VectorXd squared_mean = VectorXd::Zero(D);
    for (auto w = x.begin(); w != x.end(); ++w){
        mean += w->second;
        squared_mean += (w->second).cwiseProduct(w->second);
    }
    mean = mean / ((double) x.size());
    VectorXd sd = squared_mean - mean.cwiseProduct(mean);
    for (int i = 0; i < D; ++i){
        sd[i] = sqrt(sd[i]);
    }
    for (auto w = x.begin(); w != x.end(); ++w){
        VectorXd tmp = VectorXd::Zero(D);
        for (int i = 0; i < D; ++i){
            tmp[i] = (w->second)[i] - mean[i];
            if (sd[i] != 0)
                tmp[i] /= sd[i];
        }
        w->second = tmp;
    }
}

void initialize(){
    int count_words = 0;
    for(auto e = words.begin(); e != words.end(); ++e){
        count_words++;
        w[*e] = VectorXd::Random(D);
        bw[*e] = 0;
        grad_w[*e] = VectorXd::Zero(D);
        grad_bw[*e] = 0;
    }

    int count_contexts = 0;
    for(auto e = contexts.begin(); e != contexts.end(); ++e){
        count_contexts++;
        c[*e] = VectorXd::Random(D);
        bc[*e] = 0;
        grad_c[*e] = VectorXd::Zero(D);
        grad_bc[*e] = 0;
    }

    centralize(w);
    centralize(c);
    fprintf(stderr, "%sInitialization Completed...\n%s", KYEL, KNRM);
}

double f(size_t x){
    if (x < 100)
        return pow((x / 100.0), 0.75);
    else
        return 1.0;
}


void train(int epohs, double alpha, double lambda){
    fprintf(stderr, "%s\nTotal ephos to train = %d\n%s", KGRN, epohs, KNRM);
    fprintf(stderr, "%sInitial learning rate = %f\n%s", KGRN, alpha, KNRM);
    fprintf(stderr, "%slambda = %f\n%s", KGRN, lambda, KNRM);
    fprintf(stderr, "%sDim = %d\n%s", KGRN, D, KNRM);

    double total_loss, cost;
    VectorXd gw = VectorXd::Zero(D);
    VectorXd gc = VectorXd::Zero(D);  
    VectorXd diff = VectorXd::Zero(D); 

    VectorXd one_vect = VectorXd::Zero(D);
    for (auto i = 0; i < D; ++i)
        one_vect[i] = 1.0;

    int found_pairs = 0;

    for (int t = 0; t < epohs; ++t){
        total_loss = 0;
        found_pairs = 0;
        for(auto e = edges.begin(); e != edges.end(); ++e){
            cost = w[e->w].dot(c[e->c]) + bw[e->w] + bc[e->c] - log(e->value);
            total_loss += f(e->value) * cost * cost;

            cost *= f(e->value);
            gw = cost * c[e->c];
            gc = cost * w[e->w];

            string pair_key = e->w + "<+>" + e->c;
            if (pairs.find(pair_key) != pairs.end()){
                diff = lambda * pairs[pair_key] * (w[e->w] - c[e->c]);
                gw += diff;
                gc -= diff;
                found_pairs++;
                }

            grad_w[e->w] += gw.cwiseProduct(gw);
            grad_c[e->c] += gc.cwiseProduct(gc);
            grad_bw[e->w] += cost * cost;
            grad_bc[e->c] += cost * cost;

            w[e->w] -= alpha * gw.cwiseProduct((grad_w[e->w] + one_vect).cwiseInverse().cwiseSqrt());
            c[e->c] -= alpha * gc.cwiseProduct((grad_c[e->c] + one_vect).cwiseInverse().cwiseSqrt());

            bw[e->w] -= (alpha * cost) / sqrt(1.0 + grad_bw[e->w]);
            bc[e->c] -= (alpha * cost) / sqrt(1.0 + grad_bc[e->c]);                   
        }
        fprintf(stderr, "Itr = %d, Loss = %f, foundPairs = %d\n", t, (sqrt(total_loss) / edges.size()), found_pairs);
    }
}


void write_line(ofstream &reps_file, VectorXd vec, string label){
    reps_file << label + " ";
    for (int i = 0; i < D; ++i)
        reps_file << vec[i] << " ";
    reps_file << endl;
}

void save_model(string fname){
    ofstream reps_file;
    reps_file.open(fname);
    if (!reps_file){
        fprintf(stderr, "%sFailed to write reps to %s\n%s", KRED, KNRM, fname.c_str());
        exit(1);
    } 
    for (auto x = words.begin(); x != words.end(); ++x){
        if (contexts.find(*x) != contexts.end())
             write_line(reps_file, 0.5 * (w[*x] + c[*x]), *x);
        else
            write_line(reps_file, w[*x], *x);     
    }

    for (auto x = contexts.begin(); x != contexts.end(); ++x){
        if (words.find(*x) == words.end())
            write_line(reps_file, c[*x], *x);
    }
    reps_file.close();
}


int main(int argc, char *argv[]){
    int no_threads = 100;
    omp_set_num_threads(no_threads);
    setNbThreads(no_threads);
    initParallel(); 

    if (argc == 1) {
        fprintf(stderr, "usage: ./reps --dim=dimensionality --model=model_fname \
                                --alpha=alpha --ephos=rounds --lmda=lambda --edges=edges_fname \
                                 --pretrain=pretrained_word_vectors_file (if any) \
                                 --pairs=pairs_file_name \n"); 
        return 0;
    }
    parse_args::init(argc, argv); 
    string edges_fname = parse_args::get<string>("--edges");
    string pretrain = parse_args::get<string>("--pretrain");
    string pairs_fname = parse_args::get<string>("--pairs");

    D = parse_args::get<int>("--dim");
    int epohs = parse_args::get<int>("--epohs");
    double alpha = parse_args::get<double>("--alpha");
    string model = parse_args::get<string>("--model");
    double lambda = parse_args::get<double>("--lmda");

    
    read_edges(edges_fname, edges);
    fprintf(stderr, "%sTotal no. of target train instances = %d\n%s", KGRN, (int) edges.size(), KNRM);
    fprintf(stderr, "%sTotal no of words = %d\n%s", KGRN, (int) words.size(), KNRM);
    fprintf(stderr, "%sTotal no. of contexts = %d\n%s", KGRN, (int) contexts.size(), KNRM); 
    read_pairs(pairs_fname);
    //test_code();
    initialize();
    if (pretrain.length() > 0){
        load_word_vectors(pretrain);
    } 
    train(epohs, alpha, lambda);
    save_model(model);
    return 0;

}