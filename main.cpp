#include "neural_network.h"

int main() {
    vector<vector<double>> inputVals = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    vector<vector<double>> targetVals = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    vector<unsigned> topology = {2, 4, 1};
    Net myNet(topology);

    vector<double> resultVals;

    for (int epoch = 0; epoch < 10000; ++epoch) {
        double totalError = 0.0;

        for (size_t i = 0; i < inputVals.size(); ++i) {
            myNet.feedForward(inputVals[i]);
            myNet.backProp(targetVals[i]);
            myNet.getResults(resultVals);

            // Print progress every 1000 epochs (on first input)
            if (epoch % 1000 == 0 && i == 0) {
                cout << "Epoch #" << epoch << endl;
                showVectorVals("Inputs:", inputVals[i]);
                showVectorVals("Outputs:", resultVals);
                showVectorVals("Target:", targetVals[i]);
                cout << "Net error: " << myNet.getRecentAverageError() << "\n\n";
            }

            totalError += pow(resultVals[0] - targetVals[i][0], 2);
        }

        // Early stopping if error is small enough
        if (totalError / inputVals.size() < 0.01)
            break;
    }

    // Final evaluation
    cout << "Final result after training: \n";
    for (size_t i = 0; i < inputVals.size(); ++i) {
        myNet.feedForward(inputVals[i]);
        myNet.getResults(resultVals);
        showVectorVals("Inputs:", inputVals[i]);
        showVectorVals("Outputs:", resultVals);
        showVectorVals("Target:", targetVals[i]);
        cout << endl;
    }

    return 0;
}
