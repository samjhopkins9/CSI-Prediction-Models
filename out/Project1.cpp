
#include "../ProcessingFunctions.h"
#include "../MachineLearning.h"
#include "../MultilayerPerceptron.h"
#include "../ConvolutionalNeuralNet.h"
#include <chrono>

int main(){

    auto startTime = chrono::high_resolution_clock::now();

    DataFrame x_train = DataFrame();
    DataFrame x_test = DataFrame();
    DataFrame x_train_large = DataFrame();
    DataFrame x_test_large = DataFrame();
    DataFrame y_train_0 = DataFrame();
    DataFrame y_test_0 = DataFrame();
    DataFrame y_train_1 = DataFrame();
    DataFrame y_test_1 = DataFrame();
    
    x_train.readTxt("x_train.txt");
    x_test.readTxt("x_test.txt");
    x_train_large.readTxt("x_train_large.txt");
    x_test_large.readTxt("x_test_large.txt");
    x_train_large.readTxt_3D("x_train_large.txt", 4, 1632);
    x_test_large.readTxt_3D("x_test_large.txt", 4, 1632);
    x_train.MinMaxNormalize();
    x_test.MinMaxNormalize();
    x_train_large.MinMaxNormalize();
    x_test_large.MinMaxNormalize();
    x_train_large.MinMaxNormalize_3D();
    x_test_large.MinMaxNormalize_3D();
    // x_test_large.print_3D();
    
    // y for regression
    y_train_0.readTxt("y_train_0.txt");
    y_test_0.readTxt("y_test_0.txt");
    vector<double> y_xpos_train = y_train_0.getColumn(0);
    vector<double> y_xpos_test = y_test_0.getColumn(0);
    vector<double> y_ypos_train = y_train_0.getColumn(1);
    vector<double> y_ypos_test = y_test_0.getColumn(1);
    
    // y for classification
    y_train_1.readTxt("y_train_1.txt");
    y_test_1.readTxt("y_test_1.txt");
    vector<double> y_class_train = y_train_1.getColumn(0);
    vector<double> y_class_test = y_test_1.getColumn(0);
    
    
    
    
    // regression (ML)
    LinearRegression linearModel = LinearRegression();
    linearModel.train(x_train, y_xpos_train, 1000, 0.5);
    
    vector<double> y_xpos_pred_train = linearModel.predictions;
    double train_error = MSE(y_xpos_pred_train, y_xpos_train);
    double train_R2 = RSquared(y_xpos_pred_train, y_xpos_train);
    cout << "Linear Regression Train MSE for X: " << train_error << endl;
    cout << "Linear Regression Train R Squared for X: " << train_R2 << endl << endl;
    
    vector<double> y_xpos_pred_test = linearModel.predict(x_test);
    double test_error = MSE(y_xpos_pred_test, y_xpos_test);
    double test_R2 = RSquared(y_xpos_pred_test, y_xpos_test);
    cout << "Linear Regression Test MSE for X: " << test_error << endl;
    cout << "Linear Regression Test R Squared for X: " << test_R2 << endl << endl;
    
    linearModel.train(x_train, y_ypos_train, 1000, 0.5);
    
    vector<double> y_ypos_pred_train = linearModel.predictions;
    double train_error1 = MSE(y_ypos_pred_train, y_ypos_train);
    double train_R21 = RSquared(y_ypos_pred_train, y_ypos_train);
    cout << "Linear Regression Train MSE for Y: " << train_error1 << endl;
    cout << "Linear Regression Train R Squared for Y: " << train_R21 << endl << endl;
    
    vector<double> y_ypos_pred_test = linearModel.predict(x_test);
    double test_error1 = MSE(y_ypos_pred_test, y_ypos_test);
    double test_R21 = RSquared(y_ypos_pred_test, y_ypos_test);
    cout << "Linear Regression Test MSE for Y: " << test_error1 << endl;
    cout << "Linear Regression Test R Squared for Y: " << test_R21 << endl << endl;
    
    
    
    
    // classification (ML)
    KNN knnModel = KNN(x_train, y_class_train, 7);
    vector<double> y_pred_train = knnModel.predict(x_train);
    double train_acc = Accuracy(y_pred_train, y_class_train);
    vector<double> y_pred_test = knnModel.predict(x_test);
    double test_acc = Accuracy(y_pred_test, y_class_test);
    cout << "KNN Training Accuracy: " << train_acc << endl;
    cout << "KNN Test Accuracy: " << test_acc << endl;
    
    LogisticRegression logisticModel = LogisticRegression();
    logisticModel.train(x_train, y_class_train, 1000, 0.01);
    
    vector<double> y_pred_train1 = logisticModel.predictions;
    double train_acc1 = Accuracy(y_pred_train1, y_class_train);
    vector<double> y_pred_test1 = logisticModel.predict(x_test);
    double test_acc1 = Accuracy(y_pred_test1, y_class_test);
    cout << "Logistic Regression Training Accuracy: " << train_acc1 << endl;
    cout << "Logistic Regression Test Accuracy: " << test_acc1 << endl;
    
    
    
    /*
    // regression (MLP)
    int HL_dim = 3;
    int HL_size = 5;
    double learningRate = 0.000001;
    double clipThreshold = 6.0;
    
    MLP perceptron = MLP(x_train_large, y_xpos_train, HL_size, HL_dim, learningRate, clipThreshold);
    
    int iterations = 5;
    perceptron.train(iterations, true);
    double train_mse_perceptron = MSE(perceptron.predicted, y_xpos_train);
    double train_R2_perceptron = RSquared(perceptron.predicted, y_xpos_train);
    cout << train_mse_perceptron << endl;
    cout << train_R2_perceptron << endl;
    // perceptron.printOutput();
    */
    
    
    /*
    // regression (CNN)
    int CL_dim = 2;
    int filterSize = 150;
    int FCL_dim = 1;
    int FCL_size = 5;
    int stride = 1;
    int poolStride = 3;
    double learningRate = 0.000001;
    double clipThreshold = 6.0;
    
    CNN conv = CNN(x_train_large, y_xpos_train, CL_dim, filterSize, FCL_dim, FCL_size, stride, poolStride, learningRate, clipThreshold);
    
    int iterations = 3;
    conv.train(iterations);
    double train_MSE_conv = MSE(conv.predicted, y_xpos_train);
    double train_R2_conv = RSquared(conv.predicted, y_xpos_train);
    cout << train_MSE_conv << endl;
    cout << train_R2_conv << endl;
    */
    
    
    
    // running time
    auto endTime = chrono::high_resolution_clock::now();
    auto runTime = chrono::duration_cast<chrono::nanoseconds>(endTime - startTime);
    cout << "run time: " << runTime.count() * pow(10, -9) << " seconds" << endl;
    
    
    
    return 0;

}
