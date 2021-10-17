const { exp } = require("mathjs");

function sigmoid(x, derivative) {
  let fx = 1 / (1 + exp(-x));

  if (derivative) return fx * (1 - fx);

  return fx;
}

module.exports = {
  sigmoid,
};
