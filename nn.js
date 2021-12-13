const activation = require("./activations");
const {
  random,
  multiply,
  subtract,
  dotMultiply,
  transpose,
  add,
  mean,
  abs,
} = require("mathjs");

const EPOCH_ERROR_INTERVAL = 10000;
const EPOCHS = 200000;
const LEARNING_RATE = 0.5;

class NeuralNetwork {
  constructor({ inputNodes, hiddenNodes, outputNodes }) {
    this.inputNodes = inputNodes;
    this.hiddenNodes = hiddenNodes;
    this.outputNodes = outputNodes;

    this.epochs = EPOCHS;
    this.activation = activation.sigmoid;
    this.learningRate = LEARNING_RATE;
    this.output = 0;

    this.inputToHiddenSynapse = random(
      [this.inputNodes, this.hiddenNodes],
      -1.0,
      1.0
    );
    this.hiddenToOutputSynapse = random(
      [this.hiddenNodes, this.outputNodes],
      -1.0,
      1.0
    );
  }

  train(input, target) {
    for (let i = 0; i < this.epochs; i++) {
      const inputLayer = input;
      const hiddenLayer = multiply(inputLayer, this.inputToHiddenSynapse).map(
        (v) => this.activation(v, false)
      );
      const outputLayer = multiply(hiddenLayer, this.hiddenToOutputSynapse).map(
        (v) => this.activation(v, false)
      );

      const outputError = subtract(target, outputLayer);
      const outputDelta = dotMultiply(
        outputError,
        outputLayer.map((v) => this.activation(v, true))
      );
      const hiddenError = multiply(
        outputDelta,
        transpose(this.hiddenToOutputSynapse)
      );
      const hiddenDelta = dotMultiply(
        hiddenError,
        hiddenLayer.map((v) => this.activation(v, true))
      );

      this.hiddenToOutputSynapse = add(
        this.hiddenToOutputSynapse,
        multiply(
          transpose(hiddenLayer),
          multiply(outputDelta, this.learningRate)
        )
      );
      this.inputToHiddenSynapse = add(
        this.inputToHiddenSynapse,
        multiply(
          transpose(inputLayer),
          multiply(hiddenDelta, this.learningRate)
        )
      );
      this.output = outputLayer;

      if (i % EPOCH_ERROR_INTERVAL === 0) {
        console.error(`Error ${mean(abs(outputError))} = Epoch ${i}`);
      }
    }
  }

  predict(inputLayer) {
    const hiddenLayer = multiply(inputLayer, this.inputToHiddenSynapse).map(
      (v) => this.activation(v, false)
    );

    return multiply(hiddenLayer, this.hiddenToOutputSynapse).map((v) =>
      this.activation(v, false)
    );
  }
}

module.exports = {
  NeuralNetwork,
};
