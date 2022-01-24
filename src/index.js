import "./styles.css";
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

document.getElementById("root").innerHTML = `
<div class="App">
<h1>Tensorflow</h1>
<p>This version is a 4 layer network - input, hidden, hidden, output - which learns that 1 = 2 and 2 = 1.</p>
<p>The predict() function doesn't give exactly 2 - it comes out with around 1.9999999999952895412 - because the neural network is just a math function. But predictAndRound() will give us the correct value.</p>
<p>You can check out the outputs of the model in the console. And see Model Predictions vs Original Data for the final outcome where 1 = 2 and 2 = 1.</p>
</div>
`;

function generateTrainingData() {
  // generate 200 samples of training data
  // 100 samples of input 1 = output 2
  // 100 samples of input 2 = output 1

  let trainingData = [];

  for (let i = 0; i < 100; i++) {
    trainingData.push({
      input: 1,
      output: 2
    });
  }
  for (let i = 0; i < 100; i++) {
    trainingData.push({
      input: 2,
      output: 1
    });
  }
  return trainingData;
}

function createModel() {
  // Create a sequential model
  const model = tf.sequential();
  // Add a single input layer
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
  // two hidden layers
  model.add(tf.layers.dense({ units: 10, activation: "relu" }));
  model.add(tf.layers.dense({ units: 10, activation: "relu" }));
  // Add an output layer
  model.add(tf.layers.dense({ units: 1, useBias: true }));
  return model;
}

function convertToTensors(data) {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map((d) => d.input);
    const labels = data.map((d) => d.output);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    return {
      inputs: inputTensor,
      labels: labelTensor
    };
  });
}

function trainModel(model, inputs, labels) {
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ["mse"]
  });

  const batchSize = 32;
  const epochs = 350;

  return model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: "Training Performance" },
      ["loss", "mse"],
      { height: 200, callbacks: ["onEpochEnd"] }
    )
  });
}

function testModel(model, inputData) {
  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {
    const testValues = [1, 1, 1, 2, 2];
    const xs = tf.tensor(testValues);
    const preds = model.predict(xs.reshape([testValues.length, 1]));

    // Un-normalize the data
    return [xs.dataSync(), preds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] };
  });

  const originalPoints = inputData.map((d) => ({
    x: d.input,
    y: d.output
  }));

  tfvis.render.scatterplot(
    { name: "Model Predictions vs Original Data" },
    {
      values: [originalPoints, predictedPoints],
      series: ["original", "predicted"]
    },
    {
      xLabel: "input",
      yLabel: "output",
      height: 300,
      xAxisDomain: [0, 2.1],
      yAxisDomain: [0, 2.1]
    }
  );
}

// Train the model
const model = createModel();
tfvis.show.modelSummary({ name: "Model Summary" }, model);

// predict a single value
function predict(value) {
  const inputTensor = tf.tensor2d([value], [1, 1]);
  const preds = model.predict(inputTensor);
  return preds.dataSync()[0];
}

function predictAndRound(value) {
  return Math.round(predict(value));
}

let data = generateTrainingData();
const { inputs, labels } = convertToTensors(data);
trainModel(model, inputs, labels).then((info) => {
  testModel(model, data);
  //console.log(info);
  console.log(predict(1));
  console.log(predict(2));
  console.log(predict(2));
  console.log(predict(1));
  console.log(predictAndRound(1));
  console.log(predictAndRound(1));
  console.log(predictAndRound(2));
  console.log(predictAndRound(2));
});
