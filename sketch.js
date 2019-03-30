let video;
let features;
let knn;
let labelP;
let ready = false;
let x;
let y;
let label = 'nothing';
//let json = {}
//json.id = 0;

let string;

function setup() {
  createCanvas(320, 240);
  video = createCapture(VIDEO);
  video.size(320, 240);
  features = ml5.featureExtractor('MobileNet', modelReady);
  knn = ml5.KNNClassifier();
  labelP = createP('need training data');
  labelP.style('font-size', '32pt');
  x = width / 2;
  y = height / 2;
}

function goClassify() {
  const logits = features.infer(video);
  knn.classify(logits, function(error, result) {
    if (error) {
      console.error(error);
    } else {
      label = result.label;
      labelP.html(result.label);
      goClassify();
    }
  });
}

function keyPressed() {
  const logits = features.infer(video);
  if (key == 'm') {
    knn.addExample(logits, 'shenYun');
    console.log(logits,'shenYun');
  } else if (key == "n") {
    knn.addExample(logits, 'notShenYun');
    console.log(logits, 'notShenYun');
  } else if (key == "s") {
    //saveJSON(json, "the module.json");
    save("the module.json");
  }
}

function modelReady() {
  console.log('Model is ready!');
  // Comment back in to load your own model!
  //knn.load('Send help/the cursed ads.json', function() {
    //console.log('knn loaded');
  //});
}

function mousePressed() {
  if (knn.getNumLabels() > 0) {
    const logits = features.infer(video);
    knn.classify(logits, gotResult);
  }
}

function gotResult(error, result) {
  if (error) {
    console.error(error);
  } else {
    console.log(result);
  }
}
function draw() {
  background(0);
  fill(255);

  if (label == 'shenYun') {
    string = "They're everywhere.";
    let textsize = 32;
    textSize(textsize);
    text(string, 10, height - 100);
    textsize++; 
  } else if (label == "notShenYun") {
    string = "...";
    let textsize = 16;
    textSize(textsize);
    text(string, x / 2, height - 100);
  }

  //image(video, 0, 0);
  if (!ready && knn.getNumLabels() > 0) {
    goClassify();
    ready = true;
  }
}

// Temporary save code until ml5 version 0.2.2
const save = (knn, name) => {
  const dataset = knn.knnClassifier.getClassifierDataset();
  if (knn.mapStringToIndex.length > 0) {
    Object.keys(dataset).forEach(key => {
      if (knn.mapStringToIndex[key]) {
        dataset[key].label = knn.mapStringToIndex[key];
      }
    });
  }
  const tensors = Object.keys(dataset).map(key => {
    const t = dataset[key];
    if (t) {
      return t.dataSync();
    }
    return null;
  });
  let fileName = 'help.json';
  if (name) {
    fileName = name.endsWith('.json') ? name : `${name}.json`;
  }
  saveFile(fileName, JSON.stringify({ dataset, tensors }));
};

const saveFile = (name, data) => {
  const downloadElt = document.createElement('a');
  const blob = new Blob([data], { type: 'octet/stream' });
  const url = URL.createObjectURL(blob);
  downloadElt.setAttribute('href', url);
  downloadElt.setAttribute('download', name);
  downloadElt.style.display = 'none';
  document.body.appendChild(downloadElt);
  downloadElt.click();
  document.body.removeChild(downloadElt);
  URL.revokeObjectURL(url);
};