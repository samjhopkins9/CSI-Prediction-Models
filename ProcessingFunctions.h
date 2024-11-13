using namespace std;

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>

// Dataset reading and reshaping
class DataFrame {
    
private:
    vector<vector<double> > matrix_2D;
    vector<vector<vector<double> > > matrix_3D;
    
public:
    int m = 0;
    int n = 0;
    int m_3D = 0;
    int n_3D = 0;
    int c_3D = 0;
    
    DataFrame(){
        
        this->matrix_2D = vector<vector<double> >();
        this->matrix_3D = vector<vector<vector<double> > > ();
        
    }
    
    void readTxt(string filename) {
        ifstream file(filename);
        string line;
        
        m = 0;
        while (getline(file, line)) {
            istringstream iss(line);
            vector<double> row;
            double value;
            
            n = 0;
            while (iss >> value) {
                row.push_back(value);
                n++;
            }
            
            matrix_2D.push_back(row);
            m++;
        }
        
    }
    
    // 1632 is specific to Project 1 problem
    void readTxt_3D(string filename, int nFeatures, int nSubfeatures){
        ifstream file(filename);
        string line;
        
        m_3D = 0;
        while (getline(file, line)) {
            istringstream iss(line);
            vector<vector<double> > row;
            
            n_3D = 0;
            for (int i=0; i<nFeatures; i++){
                vector<double> r;
                double value;
                
                c_3D = 0;
                while (iss >> value) {
                    r.push_back(value);
                    c_3D++;
                    if ((c_3D % nSubfeatures) == 0){
                        row.push_back(r);
                        continue;
                    }
                }
                n_3D++;
            }
            m_3D++;
            matrix_3D.push_back(row);
        }
        
    }
    
    vector<double> getColumn(int col){
        
        vector<double> feature(m);
        for (int i=0; i<m; i++)
            feature[i] = matrix_2D[i][col];
        
        return feature;
    }
    
    vector<double> getRow(int row){
        
        return matrix_2D[row];
        
    }
    
    vector<vector<double> > getImage(int instance){
        
        return matrix_3D[instance];
        
    }
    
    void transpose(){
        vector<vector<double> > transposedMatrix(n, vector<double>(m));
        for (int i=0; i<m; i++){
            for (int j=0; j<n; j++){
                transposedMatrix[j][i] = matrix_2D[i][j];
            }
        }
        int a = m;
        m = n;
        n = a;
        matrix_2D = transposedMatrix;
        
    }
    
    
    // Dataset normalization/standardization
    void MinMaxNormalize(){
        
        for (int j=0; j<n; j++){
            
            double minVal = matrix_2D[0][j];
            double maxVal = matrix_2D[0][j];
            
            for (int i = 0; i < m; i++) {
                if (matrix_2D[i][j] < minVal)
                    minVal = matrix_2D[i][j];
                if (matrix_2D[i][j] > maxVal)
                    maxVal = matrix_2D[i][j];
            }
            
            // Normalize the j-th feature
            for (int i = 0; i < m; i++) {
                matrix_2D[i][j] = (matrix_2D[i][j] - minVal) / (maxVal - minVal);
            }
            
        }
        
    }
    
    void MinMaxNormalize_3D(){
        
        for (int j=0; j<n_3D; j++){
            
            for (int k=0; k<c_3D; k++){
                
                double minVal = matrix_3D[0][j][k];
                double maxVal = matrix_3D[0][j][k];
                for (int i=0; i<m_3D; i++){
                    
                    if (matrix_3D[i][j][k] < minVal)
                        minVal = matrix_3D[i][j][k];
                    if (matrix_3D[i][j][k] > maxVal)
                        maxVal = matrix_3D[i][j][k];
                    
                }
                
                for (int i=0; i<m_3D; i++)
                    matrix_3D[i][j][k] = (matrix_3D[i][j][k] - minVal) / (maxVal - minVal);
                
                
            }
            
        }
        
    }
    
    void print(){
        
        for (int i=0; i<m; i++){
            
            for (int j=0; j<n; j++)
                cout << matrix_2D[i][j] << " ";
            cout << endl;

        }
        
    }
    
    void print_3D(){
        
        for (int i=0; i<m; i++){
            
            for (int j=0; j<n; j++){
                for (int c=0; c<matrix_3D[i][j].size(); c++)
                    cout << matrix_3D[i][j][c] << " ";
                cout << endl;
            }

        }
        
    }
};

// functions used in regression and classification tasks
double dotProduct(vector<double> a, vector<double> b){

    if (a.size() != b.size())
        return -1;

    double s = 0;

    for (int i=0; i<a.size(); i++){
    
        s += a[i] * b[i];
    
    }
    return s;

}

double euclidianDistance(vector<double> a, vector<double> b){

    if (a.size() != b.size())
        return -1;
    
    double sum = 0;

    for (int i=0; i<a.size(); i++)
        sum += (b[i] - a[i]) * (b[i] - a[i]);
    
    
    return sqrt(sum);

}


// activation functions (and their derivatives)
double sigmoid(double z){
    return 1.0/(1.0 + exp(-z));
}

double sigmoidDerivative(double z) {
    return sigmoid(z) * (1 - sigmoid(z));
}

double relu(double z) {
    return max(0.0, z);
}

double leakyRelu(double z){

    return z > 0 ? z : 0.01*z;

}

double reluDerivative(double z) {
    return z > 0 ? 1 : 0;
}

double leakyReluDerivative(double z){

    return z > 0 ? 1 : 0.01;

}

double tanh(double z) {
    double ez = exp(z);
    double e_neg_z = exp(-z);
    return (ez - e_neg_z) / (ez + e_neg_z);
}

double tanhDerivative(double z) {
    double t = tanh(z);
    return 1 - t * t;
}


// Loss functions
double MSE(vector<double> yh, vector<double> y){

    double sum = 0;
    for (int i=0; i<y.size(); i++){
    
        sum += (y[i] - yh[i]) * (y[i] - yh[i]);
    
    }
    sum /= y.size();
    return sum;

}

double RSquared(vector<double> yh, vector<double> y){

    int n = y.size();
    double u = 0;
    for (int i=0; i<n; i++)
        u += y[i];
    u /= n;

    double sum1 = 0;
    double sum2 = 0;
    for (int i=0; i<n; i++){
    
        sum1 += (y[i] - yh[i]) * (y[i] - yh[i]);
        sum2 += (y[i] - u) * (y[i] - u);
    
    }
    // cout << sum1 << " " << sum2 << endl;
    return 1 - (sum1 / sum2);

}

double Accuracy(vector<double> y, vector<double> yh){

    double s = 0;
    for (int i=0; i<y.size(); i++){
    
        if (y[i] == yh[i])
            s++;
    
    }
    return s/y.size();

}
