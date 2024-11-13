#include <random>

class ConvolutionalLayer {
public:
        
    vector<double> filter;
    double bias;
    int stride;
    vector<vector<vector<double> > > convOutputs;
    vector<vector<vector<double> > > activationDerivatives;

    int poolStride;
    vector<vector<vector<double> > > pooledOutputs;
    
    ConvolutionalLayer(){} ConvolutionalLayer(int filterLength, int stride, int poolStride){
        
        this->stride = stride;
        this->poolStride = poolStride;
                
        // weight/bias initialization
        this->filter = vector<double>(filterLength);
        
        double sD = 0.5 * sqrt(2.0/filterLength);
        normal_distribution<> d(0, sD);
        
        for (int i=0; i<filterLength; i++){
            filter[i] = d(gen);
        }
        this->bias = d(gen);
        
        this->convOutputs = vector<vector<vector<double> > >(20000);
        this->pooledOutputs = vector<vector<vector<double> > >(20000);
        this->activationDerivatives = vector<vector<vector<double> > >(20000);
        
    }
    
    void convolve(vector<vector<double> > image, int c){
        
        pooledOutputs[c].clear();
        convOutputs[c].clear();
        
        // convolution
        int nAntenna = image.size();
        int nSubcarriers = image[0].size();
        int f = filter.size();
        int convolutionsPerAntenna = nSubcarriers - f;
        for (int i=0; i<nAntenna; i++){
            vector<double> outputRow;
            vector<double> Aprime;
                        
            for (int j=0; j<convolutionsPerAntenna; j += stride){
                
                double s = 0;
                for (int k=0; k<f; k++)
                    s += filter[k] * image[i][j+k];
                s += bias;
                outputRow.push_back(leakyRelu(s));
                Aprime.push_back(leakyReluDerivative(s));
                
            }
            convOutputs[c].push_back(outputRow);
            activationDerivatives[c].push_back(Aprime);
            
            
        }
        
        
        // pooling
        for (int i=0; i<convOutputs[c].size(); i++){
            
            vector<double> pooledOutputRow;
            for (int j=1; j<convOutputs[c][i].size()-1; j += poolStride)
                pooledOutputRow.push_back(convOutputs[c][i][j-1] > convOutputs[c][i][j+1] ? convOutputs[c][i][j-1] : convOutputs[c][i][j+1]);
                
            pooledOutputs[c].push_back(pooledOutputRow);
            
        }
        
    }
    
    vector<double> flatten(int c){
        
        vector<double> flattenedOutputs;
        for (int i=0; i<pooledOutputs[c].size(); i++){
            for (int j=0; j<pooledOutputs[c][i].size(); j++)
                flattenedOutputs.push_back(pooledOutputs[c][i][j]);
            
        }
        pooledOutputs[c].clear();
        return flattenedOutputs;
        
    }
    
};


class CNN {
private:
    DataFrame input;
    vector<vector<vector<vector<double> > > > CLInputs;
    vector<vector<vector<double> > > FCLInputs;
    
    vector<ConvolutionalLayer> CLs;
    vector<Layer> FCLs;
    Neuron outputLayer;
    
public:
    
    vector<double> predicted;
    vector<double> observed;
    
    int CL_dim;
    int filterSize;
    int FCL_dim;
    int FCL_size;
    int stride;
    int poolStride;
    double learningRate;
    double clipThreshold;
    
    CNN(DataFrame x, vector<double> y, int CL_dim, int filterSize, int FCL_dim, int FCL_size, int stride, int poolStride, double learningRate, double clipThreshold){
        
        this->input = x;
        this->predicted = vector<double>(y.size());
        this->observed = y;
        
        this->CL_dim = CL_dim;
        this->filterSize = filterSize;
        this->FCL_dim = FCL_dim;
        this->FCL_size = FCL_size;
        this->stride = stride;
        this->poolStride = poolStride;
        this->learningRate = learningRate;
        this->clipThreshold = clipThreshold;
        
        this->CLs = vector<ConvolutionalLayer>(CL_dim);
        for (int i=0; i<CL_dim; i++)
            CLs[i] = ConvolutionalLayer(filterSize, stride, poolStride);
            
        this->FCLs = vector<Layer>(FCL_dim);
        
        this->CLInputs = vector<vector<vector<vector<double> > > >(input.m, vector<vector<vector<double> > >(CL_dim));
        this->FCLInputs = vector<vector<vector<double> > >(input.m, vector<vector<double> >(FCL_dim+1));
        this->outputLayer = Neuron(FCL_size, false);
        for (int c=0; c<input.m_3D; c++)
            forwardPass(c, c == 0 ? true : false);

    }
    
    void forwardPass(int c, bool initializeLayers = false){
        
        CLInputs[c][0] = input.getImage(c);
        for (int j=0; j<CL_dim; j++){
            CLs[j].convolve(CLInputs[c][j], c);
            // cout << "Forward pass c = " << c << endl;
            if (j != CL_dim-1)
                CLInputs[c][j+1] = CLs[j].pooledOutputs[c];
            else
                FCLInputs[c][0] = CLs[j].flatten(c);
        }
        for (int j=0; j<FCL_dim; j++){
            if (initializeLayers)
                FCLs[j] = Layer(FCL_size, FCLInputs[c][j].size());

            vector<double> outputVector(FCL_size);
            for (int k=0; k<FCL_size; k++)
                outputVector[k] = FCLs[j].neurons[k].computeOutput(FCLInputs[c][j], c);
            
            FCLInputs[c][j+1] = outputVector;
            
        }
        predicted[c] = outputLayer.computeOutput(FCLInputs[c][FCL_dim], c);
        cout << predicted[c] << endl;
        
    }
    
    void backwardPass(int c){
        
        // Compute initial output error for this instance
        double output_error = predicted[c] - observed[c];

        // Initialize error vectors for each layer, starting with the output error
        vector<vector<double> > FCL_errors(FCL_dim + 1, vector<double>(FCL_size, 0.0));
        FCL_errors[FCL_dim][0] = output_error;

        
        // Step 1 Backward propagate errors through FCLs
        for (int i=FCL_dim-1; i>=0; i--) {
            // Compute error for each neuron in this layer
            for (int j=0; j<FCL_size; j++) {
                double error_sum = 0.0;
                if (i == FCL_dim-1){
                    error_sum += FCL_errors[i+1][0] * outputLayer.weights[j];
                } else {
                    for (int p = 0; p < FCL_errors[i+1].size(); p++)
                        error_sum += FCL_errors[i+1][p] * FCLs[i+1].neurons[p].weights[j];
                }
                double weightedSum = FCLs[i].neurons[j].weightedSums[c];
                FCL_errors[i][j] = error_sum * leakyReluDerivative(weightedSum);
            }
        }
        
        
        for (int l = 0; l <= FCL_dim; l++) {
        
            // gradient clipping
            for (int r=0; r<FCL_errors[l].size(); r++){
                if (abs(FCL_errors[l][r]) > clipThreshold)
                    FCL_errors[l][r] = (FCL_errors[l][r] > 0 ? 1 : -1) * clipThreshold;
            }
        
            if (l < FCL_dim){
                
                for (int j = 0; j < FCL_size; j++) {
                    
                    double gradient = FCL_errors[l][j];
        
                    // Update weights and biases
                    for (int p = 0; p < FCLs[l].neurons[j].weights.size(); p++) {
                        double neuronInput = FCLInputs[c][l][p];
                        FCLs[l].neurons[j].weights[p] -= learningRate * gradient * neuronInput;
                        // cout << "Layer " << l << " neuron " << j << " weight " << p << " update: " << learningRate * gradient * neuronInput << endl;
                    }
                    FCLs[l].neurons[j].bias -= learningRate * gradient;
                    // cout << "Layer " << l << " neuron " << j << " bias" << " update: " << learningRate * gradient << endl;
                }
            }
            else {
                double gradient = FCL_errors[l][0];
                for (int p=0; p<outputLayer.weights.size(); p++){
                    double neuronInput = FCLInputs[c][l][p];
                    outputLayer.weights[p] -= learningRate * gradient * neuronInput;
                    // cout << "Layer " << l << " neuron " << 0 << " weight " << p << " update: " << learningRate * gradient * neuronInput << endl;
                }
                outputLayer.bias -= learningRate * gradient;
                // cout << "Layer " << l << " neuron " << 0 << " bias" << " update: " << learningRate * gradient << endl;
            }
        }
        
        
        
        // Step 2: Backpropagate errors through convolutional layers
        for (int i = CL_dim - 1; i >= 0; i--) {
            
        }
        // cout << "running" << endl;
        
    }
    
    void train(int iterations){
        
        for (int i=0; i<iterations; i++){
            for (int c=0; c<input.m_3D; c++){
                
                backwardPass(c);
                forwardPass(c);
                
            }
            
            
        }
        
    }
    
    
};
