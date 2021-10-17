const { NeuralNetwork } = require("./nn");
const { matrix } = require("mathjs");

const input = matrix([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
]);
const target = matrix([[0], [1], [1], [0]]);

const neuralNetwork = new NeuralNetwork({
  inputNodes: 2,
  hiddenNodes: 2,
  outputNodes: 1,
});

neuralNetwork.train(input, target);

console.log(neuralNetwork.predict(input));
