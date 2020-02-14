/* Mimic the online prediction latency with C++ implementation
 * you may need the MKL library in real online serving cases */
#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <queue>
#include <stack>
#include <string>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <math.h>
#include <utility>      // std::pair, std::make_pair
// generate random numbers
#include <stdio.h>      // printf, scanf, puts, NULL
#include <stdlib.h>     // srand, rand
#include <time.h>       // time 

// Credit to: https://github.com/uestla/Sparse-Matrix
//#include "<libs_dir>/src/SparseMatrix/SparseMatrix.cpp"
#include "/home/deng106/work/Sparse_DeepFwFM/latency/src/SparseMatrix/SparseMatrix.cpp"

using namespace std;


typedef std::chrono::time_point<std::chrono::system_clock> MYTIME;


const int FIELD = 39;
const int EMBEDDING = 10;
const int NODES = 400;
const int LAYERS = 3;

/* feature dimension in different fields
 * the first 13 fileds are numerical features of dimension 1 */
vector<int> dim{1,1,1,1,1,1,1,1,1,1,1,1,1, 1458, 556, 245197, 166166, 306, 20, 12055, 634, 4, 46330, 5229, 243454, 3177, 27, \
            11745, 225322, 11, 4727, 2058, 5, 238640, 18, 16, 67856, 89, 50942};

/* Logistic regression, complexity O(FIELD)
 * dim Xi: FIELD
 * dim Xv: FIELD
 * dim linear: FIELD X DIM */
float LR(int y, vector<int> & Xi, vector<float> & Xv, float bias, vector<vector<float>> & linear) {
    float sum = bias;
    for (int i = 0; i < FIELD; i++) {
	sum += linear[i][Xi[i]] * Xv[i];
    }
    return log(1 + exp(-y * sum));
}

/* Factorization machines, complexity O(FIELD X EMBEDDING)
 * dim Xi: FIELD
 * dim Xv: FIELD
 * dim linear: FIELD X DIM
 * dim quadratic: FIELD X DIM X EMBEDDING */
float FM(int y, vector<int> & Xi, vector<float> & Xv, float bias, vector<vector<float>> & linear, vector<vector<vector<float>>> & quadratic, bool pred_type=true) {
    float sum = bias;
    for (int i = 0; i < FIELD; i++) {
        sum += linear[i][Xi[i]] * Xv[i];
    }
    for (int k = 0; k < EMBEDDING; k++) {
	float sum_squared = 0;
	float squared_sum = 0;
        for (int i = 0; i < FIELD; i++) {
	    float item = quadratic[i][Xi[i]][k] * Xv[i];
	    squared_sum += item;
	    sum_squared += (item * item);
	}
	squared_sum = squared_sum * squared_sum;
	sum += 0.5 * (squared_sum - sum_squared);
    }
    if (pred_type == true)
	return log(1 + exp(-y * sum));
    else
	return sum;
}

/* Field weighted Factorization machines, complexity O(FIELD X FIELD X EMBEDDING)
 * dim Xi: FIELD
 * dim Xv: FIELD
 * dim linear: FIELD X DIM
 * dim quadratic: FIELD X DIM X EMBEDDING 
 * dim corr: FIELD X FIELD */
float FwFM(int y, vector<int> & Xi, vector<float> & Xv, float bias, vector<vector<float>> & linear, \
		vector<vector<vector<float>>> & quadratic, vector<vector<float>>& corr, bool pred_type=true) {
    float sum = bias;
    for (int i = 0; i < FIELD; i++) {
        sum += linear[i][Xi[i]] * Xv[i];
    }
    for (int i = 0; i < FIELD; i++) {
        for (int j = i + 1; j < FIELD; j++) {
	    for (int k = 0; k < EMBEDDING; k++) {
	        sum += quadratic[i][Xi[i]][k] * quadratic[j][Xi[j]][k] * corr[i][j];
	    }
        }
    }
    if (pred_type == true)
        return log(1 + exp(-y * sum));
    else
        return sum;
}

/* Deep component in DeepFM or DeepFwFM, complexity O(LAYERS X NEURONS X NEURONS)
 * dim Xi: FIELD
 * dim quadratic: FIELD X DIM X EMBEDDING
 * dim linear_1_weight: NODES X (FIELD X EMBEDDINS)
 * dim linear_weights: (LAYERS - 1) X NODES X NODES
 * dim neurons: LAYERS x NODES
 * dim bias: (LAYERS + 1) X NODES */
float Dense_DNN(int y, float output, vector<int> & Xi, vector<vector<vector<float>>> & quadratic, vector<vector<vector<float>>> & linear_weights, \
		vector<vector<float>> & linear_1_weight, vector<vector<float>> & bias, vector<vector<float>> & neurons) {
    float sum = output;
    vector<float> embedding_combined_vector;
    for (int f = 0; f < FIELD; f++) {
        for (int k = 0; k < EMBEDDING; k++) {
	    embedding_combined_vector.push_back(quadratic[f][Xi[f]][k]);
        }	
    }
    // Deal with the first layer
    for (int n = 0; n < NODES; n++) {
        for (int i = 0; i < FIELD * EMBEDDING; i++) {
            neurons[0][n] +=  embedding_combined_vector[i] * linear_1_weight[n][i] + bias[0][n];
        }
    }
    
    // Deal with hiden layers
    for (int l = 1; l < LAYERS; l++) {
	for (int n = 0; n < NODES; n++) {
            for (int k = 0; k < NODES; k++) {
                neurons[l][n] += neurons[l-1][k] * linear_weights[l-1][n][k] + bias[l][n];
            }
        }
    }
    
    // Deal with the last layer
    for (int n = 0; n < NODES; n++) {
	sum += neurons[LAYERS-1][n] * bias[LAYERS][n];
    }
    return log(1 + exp(-y * sum));
}

float Sparse_DNN(int y, float output, vector<int> & Xi, vector<vector<vector<float>>> & quadratic, vector<SparseMatrix<float>> & slinear_weights, \
		SparseMatrix<float> & slinear_1_weight, vector<vector<float>> & bias, vector<vector<float>> & neurons) {
    float sum = output;
    vector<float> embedding_combined_vector;
    for (int f = 0; f < FIELD; f++) {
        for (int k = 0; k < EMBEDDING; k++) {
            embedding_combined_vector.push_back(quadratic[f][Xi[f]][k]);
        }
    }
    // Deal with the first layer
    neurons[0] = slinear_1_weight * embedding_combined_vector;
    for (int n = 0; n < NODES; n++) {
	neurons[0][n] += bias[0][n];
    }
    // Deal with the hidden layer
    for (int l = 1; l < LAYERS; l++) {
	neurons[l] = slinear_weights[l - 1] * neurons[l - 1];
	for (int n = 0; n < NODES; n++) {
            neurons[l][n] += bias[l][n];
    	}
    }
    // Deal with the last layer
    for (int n = 0; n < NODES; n++) {
        sum += neurons[LAYERS - 1][n] * bias[LAYERS][n];
    }
    return log(1 + exp(-y * sum));
}

/* Initialize the weight in the sparse DNN component */
SparseMatrix<float> init_sparse_mat(int m, int n, float ratio) {
    SparseMatrix<float> mat(m, n);
    int nonzeros = ratio * m * n;
    int rand_i, rand_j;
    float rand_v;
    
    set<pair<int, int>> st;
    // random seed, if you comment out this one, no randomness will be given.
    srand (time(NULL));
    while (1) {
        rand_i = rand() % m + 1;
        rand_j = rand() % n + 1;
	rand_v = float(rand() % 1000) / 1000;
	//cout << rand_v << endl;
	//cout << rand_i << " " << rand_j << "set size: "<<  st.size() <<  " target numbers " << nonzeros << " total " << m * n <<  endl;
        mat.set(rand_v, rand_i, rand_j);
	st.insert(make_pair(rand_i, rand_j));
	if (st.size() == nonzeros) {
	    break;
	}
    }

    return mat;
}

/* Initialize the weights of FM component in DeepFM or DeepFwFM
 * dim linear: FIELD X DIM
 * dim quadratic: FIELD X DIM X EMBEDDING */
void init_FM(vector<vector<float>>&linear, vector<vector<vector<float>>>& quadratic) {
    for (int i = 0; i < FIELD; i++) {
        vector<float> mp_linear;
	vector<vector<float>> mp_quadratic;
	for (int j = 0; j < dim[i]; j++) {
	    mp_linear.push_back(j * j * 1.11);
	    mp_quadratic.push_back(vector<float>(EMBEDDING, j * 1.2));
	}
	linear.push_back(mp_linear);
	quadratic.push_back(mp_quadratic);
    }
}

void init_sparse_deep(SparseMatrix<float> & slinear_1_weight, vector<SparseMatrix<float>> & slinear_weights, float ratio) {
    slinear_1_weight = init_sparse_mat(NODES, EMBEDDING * FIELD, ratio);
    slinear_weights.clear();
    for (int l = 1; l < LAYERS; l++) {
	slinear_weights.push_back(init_sparse_mat(NODES, NODES, ratio));
    }
}

double cnt_time(MYTIME start, MYTIME end) {
    return (end - start).count() * ((double) std::chrono::high_resolution_clock::period::num \
		    / std::chrono::high_resolution_clock::period::den);
}

int main() {
    SparseMatrix<float> mat = init_sparse_mat(NODES, NODES, 0.01);
    vector<float> vec(NODES, 2);
    // A prediction sample
    vector<int> Xi{0,0,0,0,0,0,0,0,0,0,0,0,0,10,10,10,10,10,10,10,10,3,10,10,10,10,10,10,10,5,10,10,4,10,10,10,100,10,10};
    vector<float> Xv{1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1, 13.1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    vector<vector<float>> linear;
    vector<vector<vector<float>>> quadratic;
    int label = 1;
    int global_bias = 0.3;
    float output = 0, pred = 0;

    // Initialize FM weights
    init_FM(linear, quadratic);
    // Initialize dense DNN weights
    vector<vector<float>> linear_1_weight(NODES, vector<float>(FIELD * EMBEDDING, 0.67));
    vector<vector<vector<float>>> linear_weights(LAYERS - 1, vector<vector<float>>(NODES, vector<float>(NODES, 1.11)));
    vector<vector<float>> neurons(LAYERS, vector<float>(NODES, 0));
    vector<vector<float>> bias(LAYERS + 1, vector<float>(NODES, 0.31));
    // Initialize sparse DNN weights
    SparseMatrix<float> slinear_1_weight(NODES, FIELD * EMBEDDING);
    vector<SparseMatrix<float>> slinear_weights;

    // Logistic regression
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 1; i < 1000; i++) {
	LR(label, Xi, Xv, global_bias, linear);
    }
    auto end = std::chrono::high_resolution_clock::now();
    printf("LR Time, ms: %.5f.\n", cnt_time(start, end));

    // Factorization machines
    start = std::chrono::high_resolution_clock::now();
    for (int i = 1; i < 1000; i++) {
	FM(label, Xi, Xv, global_bias, linear, quadratic);
    }
    end = std::chrono::high_resolution_clock::now();
    printf("FM Time, ms: %.5f.\n", cnt_time(start, end));

    // FwFM
    vector<vector<float>> corr(FIELD, vector<float>(FIELD, 1.0));
    start = std::chrono::high_resolution_clock::now();
    for (int i = 1; i < 1000; i++) {
        FwFM(label, Xi, Xv, global_bias, linear, quadratic, corr);
    }
    end = std::chrono::high_resolution_clock::now();
    printf("FwFM Time, ms: %.5f.\n", cnt_time(start, end));

    // DeepFM sparse
    init_sparse_deep(slinear_1_weight, slinear_weights, 0.01);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 1; i < 1000; i++) {
	output = FM(label, Xi, Xv, global_bias, linear, quadratic, false);
        pred = Sparse_DNN(label, output, Xi, quadratic, slinear_weights, slinear_1_weight, bias, neurons);
    }
    end = std::chrono::high_resolution_clock::now();
    printf("Sparse DeepFM Time, ms: %.3f,  sparse rate: %.1f%%\n", cnt_time(start, end), 100 - 0.01 * 100);

    
    // Sparse DeepFwFM
    float list_rates[]= {0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 1.0};
    set<float> rates(list_rates, list_rates+7);
    for (auto sparse_r: rates) {
	init_sparse_deep(slinear_1_weight, slinear_weights, sparse_r);
    	start = std::chrono::high_resolution_clock::now();
	for (int i = 1; i < 1000; i++) {
	    output = FwFM(label, Xi, Xv, global_bias, linear, quadratic, corr, false);
	    pred = Sparse_DNN(label, output, Xi, quadratic, slinear_weights, slinear_1_weight, bias, neurons);
	}
	end = std::chrono::high_resolution_clock::now();    
	printf("Sparse DeepFwFM Time, ms: %.3f,  sparsity: %.1f%%\n", cnt_time(start, end), 100 - sparse_r * 100);
    }

    /*
    // Dense DeepFwFM
    start = std::chrono::high_resolution_clock::now();
    for (int i = 1; i < 1000; i++) {
	output = FwFM(label, Xi, Xv, global_bias, linear, quadratic, corr);
        pred = Dense_DNN(label, output, Xi, quadratic, linear_weights, linear_1_weight, bias, neurons);
    }
    end = std::chrono::high_resolution_clock::now();
    printf("Dense deep computation time, ms: %.3f.", cnt_time(start, end));*/
    
    return 0;
}
