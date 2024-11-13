#include <random>

mt19937 gen(static_cast<unsigned>(std::time(nullptr)));
class Neuron {
public:
    vector<double> weights;
    double bias;
    bool activate;
    
    vector<double> weightedSums;
    vector<double> outputs;

    Neuron(){} Neuron(int inputDim, bool activate = true){
    
        this->weights = vector<double>(inputDim);
        
        double sD = 0.5 * sqrt(2.0/inputDim);
        normal_distribution<> d(0, sD);
        
        for (int i=0; i<inputDim; i++){
            weights[i] = d(gen);
        }
        this->bias = d(gen);
        this->activate = activate;
        
        this->outputs = vector<double>(20000);
        this->weightedSums = vector<double>(20000);
    
    }
    
    double computeOutput(vector<double> inputs, int c){
        double linearComb = dotProduct(inputs, weights) + bias;
        weightedSums[c] = linearComb;
        double output;
        if (activate)
            output = leakyRelu(linearComb);
        else
            output = linearComb;
        outputs[c] = output;
        return output;
    }

};

class Layer {
public:
    vector<Neuron> neurons;
    int layer_size;
    Layer(){} Layer(int nNeurons, int inputDim){
    
        this->neurons = vector<Neuron>(nNeurons);
        for (int i=0; i<nNeurons; i++){
            neurons[i] = Neuron(inputDim);
        }
        this->layer_size = nNeurons;
    }


};

class MLP {

private:
    DataFrame input;
    vector<vector<vector<double> > > LayerInputs;
    vector<Layer> hiddenLayers;
    Neuron outputLayer;
public:
    
    vector<double> predicted;
    vector<double> observed;
    
    int HL_size;
    int HL_dim;
    double learningRate;
    double clipThreshold;
    
    MLP(DataFrame x, vector<double> y, int HL_size, int HL_dim, double learningRate, double clipThreshold){
        
        this->input = x;
        this->predicted = vector<double>(x.m);
        this->observed = y;
        
        this->HL_size = HL_size;
        this->HL_dim = HL_dim;
        this->learningRate = learningRate;
        this->clipThreshold = clipThreshold;
        
        this->hiddenLayers = vector<Layer>(HL_dim);
        for (int i=0; i<HL_dim; i++){
            hiddenLayers[i] = Layer(HL_size, i==0 ? x.n : HL_size);
        }
        this->outputLayer = Neuron(HL_size, false);
        
        vector<vector<double> > phVector(HL_dim+1);
        for (int i=0; i<=HL_dim; i++){
            phVector[i] = vector<double>(i == 0 ? x.n : HL_size);
        }
        this->LayerInputs = vector<vector<vector<double> > >(x.m, phVector);
    
        for (int c=0; c<x.m; c++)
            forward_pass(c);
    }
    
    void forward_pass(int c, bool verbose = false){
            LayerInputs[c][0] = input.getRow(c);
            for (int i=1; i<=HL_dim; i++){
                                
                vector<double> neuronOutputs;
                for (int j=0; j<HL_size; j++){
                    double output = hiddenLayers[i-1].neurons[j].computeOutput(LayerInputs[c][i-1], c);
                    neuronOutputs.push_back(output);
                    if (verbose)
                        cout << "Layer " << i-1 << " Neuron " << j << " Output: " << output << endl;
                }
            
                LayerInputs[c][i] = neuronOutputs;
        
            }
            predicted[c] = outputLayer.computeOutput(LayerInputs[c][HL_dim], c);
            if(verbose)
                cout << "Predicted: " << predicted[c] << endl;
    }
    
    void backward_pass(int c) {
            // Compute initial output error for this instance
            double output_error = predicted[c] - observed[c]; // ¥ - Y

            // Initialize error vectors for each layer, starting with the output error
            vector<vector<double> > errors(HL_dim + 1, vector<double>(HL_size, 0.0));
            errors[HL_dim][0] = output_error;

            // Step 1 Backward propagate errors
            for (int i=HL_dim-1; i>=0; i--) {
                // Compute error for each neuron in this layer
                for (int j=0; j<HL_size; j++) {
                    double error_sum = 0.0; // ∑∂L/∂øij
                    if (i == HL_dim-1){
                        error_sum += errors[i+1][0] * outputLayer.weights[j];
                    } else {
                        for (int p = 0; p < errors[i+1].size(); p++) {
                            error_sum += errors[i+1][p] * hiddenLayers[i+1].neurons[p].weights[j];
                        }
                    }
                    double weightedSum = hiddenLayers[i].neurons[j].weightedSums[c];
                    errors[i][j] = error_sum * leakyReluDerivative(weightedSum); // ∂L/∂ø(ij) * ∂ø(ij)/∂Z(ij) = ∂L/∂Z(ij)
                }
            }
            // cout << "running" << endl;

            // Step 2 Update weights and biases
            for (int l = 0; l <= HL_dim; l++) {
            
                // gradient clipping
                for (int r=0; r<errors[l].size(); r++){
                    if (abs(errors[l][r]) > clipThreshold)
                        errors[l][r] = (errors[l][r] > 0 ? 1 : -1) * clipThreshold;
                }
            
                if (l < HL_dim){
                    
                    for (int j = 0; j < HL_size; j++) {
                        
                        double gradient = errors[l][j];
            
                        // Update weights and biases
                        for (int p = 0; p < hiddenLayers[l].neurons[j].weights.size(); p++) {
                            double neuronInput = LayerInputs[c][l][p];
                            hiddenLayers[l].neurons[j].weights[p] -= learningRate * gradient * neuronInput; // -= ∂L/∂Z(ij) * ∂Z(ij)/∂W(ij)p
                            
                            // cout << "Layer " << l << " neuron " << j << " weight " << p << " update: " << learningRate * gradient * neuronInput << endl;
                        }
                        hiddenLayers[l].neurons[j].bias -= learningRate * gradient; // -= ∂L/∂Z(ij)
                        
                        // cout << "Layer " << l << " neuron " << j << " bias" << " update: " << learningRate * gradient << endl;
                    }
                }
                else {
                    double gradient = errors[l][0];
                    for (int p=0; p<outputLayer.weights.size(); p++){
                        double neuronInput = LayerInputs[c][l][p];
                        outputLayer.weights[p] -= learningRate * gradient * neuronInput;
                        // cout << "Layer " << l << " neuron " << 0 << " weight " << p << " update: " << learningRate * gradient * neuronInput << endl;
                    }
                    outputLayer.bias -= learningRate * gradient;
                    // cout << "Layer " << l << " neuron " << 0 << " bias" << " update: " << learningRate * gradient << endl;
                }
            }
    }

    
    void train(int n, bool verbose = false){
    
        for (int i=0; i<n; i++){
            for (int c=0; c<input.m; c++){
                backward_pass(c);
                forward_pass(c, verbose);
                // cout << endl << endl;
            }
        }
    
    }
    
    void printOutput(){
    
        for (int i=0; i<predicted.size(); i++){
            cout << predicted[i] << " " << observed[i] << endl;
        }
    
    }

};
