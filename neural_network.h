#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

struct Connection
{
    double weight;       // Weight of the connection
    double deltaWeight;  // Change applied to weight during last update (for momentum)
};

class Neuron;

typedef vector<Neuron> Layer;

class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    // Training parameters
    static double eta;    // Learning rate (how big steps are in gradient descent)
    static double alpha;  // Momentum (fraction of previous Δweight kept)

    // Activation function and derivative
    static double transferFunction(double x); 
    static double transferFunctionDerivative(double x);

    static double randomWeight(void) { return rand() / double(RAND_MAX); }

    // Sum of derivatives of weights * gradients from the next layer
    double sumDOW(const Layer &nextLayer) const;

    double m_outputVal;              // Output value of this neuron
    vector<Connection> m_outputWeights; // Weights going out of this neuron
    unsigned m_myIndex;              // Index of this neuron in its layer
    double m_gradient;               // Gradient (used in backprop)
};

double Neuron::eta = 0.15;  
double Neuron::alpha = 0.5;   

// ------------------- Neuron methods -------------------

void Neuron::updateInputWeights(Layer &prevLayer)
{
    // Update weights coming *into this neuron* using gradient descent
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        // New Δw = η * outputVal(prev neuron) * gradient(this neuron) + α * oldΔw
        double newDeltaWeight =
            eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;

        // Update stored Δw
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        // Apply weight update
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    // Sum over: weight(this->nextNeuron) * gradient(nextNeuron)
    // Used when computing hidden layer gradients
    double sum = 0.0;

    // Skip bias neuron in next layer (last one)
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    // For hidden neurons:
    // Gradient = Σ (w * grad_next) * f’(output)
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
    // For output neurons:
    // Gradient = (target - output) * f’(output)
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

// ---- Activation function: tanh ----
// f(x) = tanh(x) ∈ [-1, 1]
double Neuron::transferFunction(double x)
{
    return tanh(x);
}

// Derivative: f’(x) = 1 - f(x)^2
// Here x is already tanh(x) because we store outputVal
double Neuron::transferFunctionDerivative(double x)
{
    return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer)
{
    // Weighted sum of outputs from prev layer neurons
    double sum = 0.0;
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() *
               prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    // Apply activation function
    m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    // Initialize each outgoing connection with a random weight
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
}

// ------------------- Net class -------------------

class Net
{
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }

private:
    vector<Layer> m_layers; 
    double m_error;  // RMS error of current training pass
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;
};

double Net::m_recentAverageSmoothingFactor = 100.0;

void Net::getResults(vector<double> &resultVals) const
{
    resultVals.clear();
    // Copy outputs from last layer (excluding bias neuron)
    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

void Net::backProp(const vector<double> &targetVals)
{
    // ---- Calculate RMS Error ----
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1; // average
    m_error = sqrt(m_error);           // RMS

    // Running average of error (for reporting)
    m_recentAverageError =
        (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
        / (m_recentAverageSmoothingFactor + 1.0);

    // ---- Calculate output gradients ----
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // ---- Calculate hidden gradients (backprop) ----
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // ---- Update weights for all layers ----
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::feedForward(const vector<double> &inputVals)
{
    // Assign input values to input neurons
    assert(inputVals.size() == m_layers[0].size() - 1);

    for (unsigned i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // Forward propagate through rest of network
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

Net::Net(const vector<unsigned> &topology)
{
    // Build layers of neurons according to topology
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        unsigned numOutputs = (layerNum == topology.size() - 1) ? 0 : topology[layerNum + 1];

        // Add neurons to the layer (+1 bias neuron)
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
        }

        // Force the bias node’s output to 1.0
        m_layers.back().back().setOutputVal(1.0);
    }
}

// Utility to print vectors nicely
void showVectorVals(string label, const vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }
    cout << endl;
}

#endif
