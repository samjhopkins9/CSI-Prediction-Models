#include <random>

// Regression
class LinearRegression {
public:
    vector<double> weights;
    double bias = 0.0;
    
    vector<double> predictions;
    
    LinearRegression(){
        
        this->predictions = vector<double>();
        this->weights = vector<double>();
        
    }
    
    void train(DataFrame x, vector<double> y, int iterations, double learningRate){
        
        this->predictions = vector<double>(x.m, 0.0);
        
        this->weights = vector<double>(x.n, 0.0);
        for (int i=0; i<x.n; i++){
            weights[i] = ((double) rand() / RAND_MAX) * 0.01;
        }
        bias = ((double) rand() / RAND_MAX) * 0.01;
        
        for (int iter = 0; iter<iterations; iter++){
        
            vector<double> weightGradients(x.n, 0.0);
            double biasGradient = 0.0;
            double totalLoss = 0.0;
        
            for (int i=0; i<x.m; i++){
            
                vector<double> sample = x.getRow(i);
                double dotprod = dotProduct(weights, sample) + bias;
                
                double error = dotprod - y[i];
                for (int j=0; j<x.n; j++){
                    weightGradients[j] += (1.0 / x.m) * error * sample[j]; // Gradient for weight[j]
                }
                biasGradient += (1.0 / x.m) * error; // Gradient for bias
                totalLoss += error * error;
            }

            // Update weights and bias
            for (int j = 0; j < x.n; j++) {
                weights[j] -= learningRate * weightGradients[j];
            }
            bias -= learningRate * biasGradient;
            totalLoss /= x.m;
            
            if (iter % 100 == 0)
                cout << "total loss at iteration " << iter << ": " << totalLoss << endl;
        }

        // Return predictions for the final weights and bias
        for (int i = 0; i < x.m; i++) {
            predictions[i] = dotProduct(weights, x.getRow(i)) + bias;
        }
        
    }
    
    
    vector<double> predict(DataFrame x){
        
        vector<double> testPredictions(x.m);
        for (int i = 0; i < x.m; i++) {
            testPredictions[i] = dotProduct(weights, x.getRow(i)) + bias;
        }
        return testPredictions;
        
    }
    
    
};



class LogisticRegression {
public:
    
    vector<double> weights;
    double bias = 0.0;
    
    vector<double> predictions;
    
    LogisticRegression(){

        weights = vector<double>();
        predictions = vector<double>();
        
    }
    
    void train(DataFrame x, vector<double> y, int iterations, double learningRate){
        
        predictions = vector<double>(x.m, 0.0);
        weights = vector<double>(x.n, 1.0);
        for (int i=0; i<x.n; i++){
            weights[i] = ((double) rand() / RAND_MAX) * 0.01;
        }
        double bias = ((double) rand() / RAND_MAX) * 0.01;
        
        for (int iter=0; iter<iterations; iter++){
        
            vector<double> weightGradients(x.n, 0.0);
            double biasGradient = 0.0;
            double totalLoss = 0.0;
        
            for (int i=0; i<x.m; i++){
                
                vector<double> sample = x.getRow(i);
                double dotprod = dotProduct(sample, weights) + bias;
                double b = sigmoid(dotprod);
                
                // calculate error and gradients
                double error = b - y[i];
                for (int j=0; j<x.n; j++){
                    weightGradients[j] += error * sample[j];
                }
                biasGradient += error;
                
                // cross-entropy loss
                totalLoss += -(y[i] * log(b) + (1 - y[i]) * log(1 - b));
            
            }
            for (int j = 0; j < x.n; j++) {
                weights[j] -= learningRate * weightGradients[j] / x.m;
            }
            bias -= learningRate * biasGradient / x.m;
                
            if (iter % 100 == 0) {
                cout << "Iteration " << iter << " - Loss: " << totalLoss / x.m << endl;
            }
        
        }
        
        for (int i=0; i<x.m; i++){
        
            double dotprod = dotProduct(x.getRow(i), weights) + bias;
            double b = sigmoid(dotprod);
            if (b >= 0.5)
                predictions[i] = 1;
            else
                predictions[i] = 0;
        
        }
        
    }
    
    
    vector<double> predict(DataFrame x){
        
        vector<double> testPredictions(x.m, 0.0);
        
        for (int i=0; i<x.m; i++){
        
            double dotprod = dotProduct(weights, x.getRow(i)) + bias;
            double b = sigmoid(dotprod);
            if (b >= 0.5)
                testPredictions[i] = 1;
            else
                testPredictions[i] = 0;
        
        }
        
        return testPredictions;
        
    }
    
    
};



class KNN {
public:
    int k;
    DataFrame trainingSet;
    vector<double> classes;
    
    KNN(DataFrame x, vector<double> y, int k){
        this->trainingSet = x;
        this->classes = y;
        this->k = k;
    }
    
    vector<double> predict(DataFrame x){
        vector<double> testPredictions(x.m);
        for (int i=0; i<x.m; i++){
        
            vector<pair<double, int>> distances;
            for (int a = 0; a < trainingSet.m; a++) {
                double eDistance = euclidianDistance(x.getRow(i), trainingSet.getRow(a));
                distances.push_back(make_pair(eDistance, a));
            }
            sort(distances.begin(), distances.end());

            double zero_votes = 0.0;
            double one_votes = 0.0;
            for (int j = 0; j < k; j++) {
                int neighborIndex = distances[j].second;
                if (classes[neighborIndex] == 0.0)
                    zero_votes++;
                else
                    one_votes++;
            }
            if (zero_votes > one_votes)
                testPredictions[i] = 0;
            else if (zero_votes == one_votes){
                if (classes[distances[k].second] == 0.0)
                    testPredictions[i] = 0;
                else
                    testPredictions[i] = 1;
            }
            else
                testPredictions[i] = 1;
            
        
        }
        return testPredictions;
    }
    
};
