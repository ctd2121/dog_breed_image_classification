CLASSES = {
 0:'Afghan Hound',
 1:'African Hunting Dog',
 2:'Airedale',
 3:'American Staffordshire Terrier',
 4:'Appenzeller',
 5:'Australian Terrier',
 6:'Bedlington Terrier',
 7:'Bernese Mountain Dog',
 8:'Blenheim Spaniel',
 9:'Border Collie',
 10:'Border Terrier',
 11:'Boston Bull',
 12:'Bouvier Des Flandres',
 13:'Brabancon Griffon',
 14:'Brittany Spaniel',
 15:'Cardigan',
 16:'Chesapeake Bay Retriever',
 17:'Chihuahua',
 18:'Dandie Dinmont',
 19:'Doberman',
 20:'English Foxhound',
 21:'English Setter',
 22:'English Springer',
 23:'Entlebucher',
 24:'Eskimo Dog',
 25:'French Bulldog',
 26:'German Shepherd',
 27:'German Short-haired Pointer',
 28:'Gordon Setter',
 29:'Great Dane',
 30:'Great Pyrenees',
 31:'Greater Swiss Mountain Dog',
 32:'Ibizan Hound',
 33:'Irish Setter',
 34:'Irish Terrier',
 35:'Irish Water Spaniel',
 36:'Irish Wolfhound',
 37:'Italian Greyhound',
 38:'Japanese Spaniel',
 39:'Kerry Blue Terrier',
 40:'Labrador Retriever',
 41:'Lakeland Terrier',
 42:'Leonberg',
 43:'Lhasa',
 44:'Maltese Dog',
 45:'Mexican Hairless',
 46:'Newfoundland',
 47:'Norfolk Terrier',
 48:'Norwegian Elkhound',
 49:'Norwich Terrier',
 50:'Old English Sheepdog',
 51:'Pekinese',
 52:'Pembroke',
 53:'Pomeranian',
 54:'Rhodesian Ridgeback',
 55:'Rottweiler',
 56:'Saint Bernard',
 57:'Saluki',
 58:'Samoyed',
 59:'Scotch Terrier',
 60:'Scottish Deerhound',
 61:'Sealyham Terrier',
 62:'Shetland Sheepdog',
 63:'Shih-Tzu',
 64:'Siberian Husky',
 65:'Staffordshire Bullterrier',
 66:'Sussex Spaniel',
 67:'Tibetan Mastiff',
 68:'Tibetan Terrier',
 69:'Walker Hound',
 70:'Weimaraner',
 71:'Welsh Springer Spaniel',
 72:'West Highland White Terrier',
 73:'Yorkshire Terrier',
 74:'Affenpinscher',
 75:'Basenji',
 76:'Basset',
 77:'Beagle',
 78:'Black-and-Tan Coonhound',
 79:'Bloodhound',
 80:'Bluetick',
 81:'Borzoi',
 82:'Boxer',
 83:'Briard',
 84:'Bull Mastiff',
 85:'Cairn',
 86:'Chow',
 87:'Clumber',
 88:'Cocker Spaniel',
 89:'Collie',
 90:'Curly-Coated Retriever',
 91:'Dhole',
 92:'Dingo',
 93:'Flat-Coated Retriever',
 94:'Giant Schnauzer',
 95:'Golden Retriever',
 96:'Groenendael',
 97:'Keeshond',
 98:'Kelpie',
 99:'Komondor',
 100:'Kuvasz',
 101:'Malamute',
 102:'Malinois',
 103:'Miniature Pinscher',
 104:'Miniature Poodle',
 105:'Miniature Schnauzer',
 106:'Otterhound',
 107:'Papillon',
 108:'Pug',
 109:'Redbone',
 110:'Schipperke',
 111:'Silky Terrier',
 112:'Soft Coated Wheaten Terrier',
 113:'Standard Poodle',
 114:'Standard Schnauzer',
 115:'Toy Poodle',
 116:'Toy Terrier',
 117:'Vizsla',
 118:'Whippet',
 119:'Wire-Haired Fox Terrier'
};

const MODEL_PATH =
    'model.json';

function sum(vals){
  var tot_sum = 0;
  for(var i = 0; i < vals.length; i++) {
    tot_sum += Number(vals[i]);
  }
  return tot_sum;
}

const IMAGE_SIZE = 192;
const TOPK_PREDICTIONS = 3;

let my_model;
const demo = async () => {

  my_model = await tf.loadLayersModel(MODEL_PATH);

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  my_model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  // Make a prediction through the locally hosted cat.jpg.
  const catElement = document.getElementById('cat');
  if (catElement.complete && catElement.naturalHeight !== 0) {
    predict(catElement);
    catElement.style.display = '';
  } else {
    catElement.onload = () => {
      predict(catElement);
      catElement.style.display = '';
    }
  }

  document.getElementById('file-container').style.display = '';
};

/**
 * Given an image element, makes a prediction through my_model returning the
 * probabilities of the top K classes.
 */
async function predict(imgElement) {

  // The first start time includes the time it takes to extract the image
  // from the HTML and preprocess it, in additon to the predict() call.
  const startTime1 = performance.now();
  // The second start time excludes the extraction and preprocessing and
  // includes only the predict() call.
  let startTime2;
  const logits = tf.tidy(() => {
    // tf.browser.fromPixels() returns a Tensor from an image element.
    const img = tf.browser.fromPixels(imgElement).toFloat();

    // const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].
    // const normalized = img.sub(offset).div(offset);
    const normalized = img.div(255.0);

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    startTime2 = performance.now();
    // Make a prediction through my_model.
    return my_model.predict(batched);
  });

  // Convert logits to probabilities and class names.
  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;

  // Show the classes in the DOM.
  showResults(imgElement, classes);
}

/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from my_model.
 * @param topK The number of top predictions to show.
 */

async function getTopKClasses(logits, topK) {
  const values = await logits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: CLASSES[topkIndices[i]],
      probability: Math.round(topkValues[i] / sum(topkValues) * 100, 1)
    })
  }
  return topClassesAndProbs;
}

//
// UI
//

function showResults(imgElement, classes) {
  const predictionContainer = document.createElement('div');
  predictionContainer.className = 'pred-container';

  const imgContainer = document.createElement('div');
  imgContainer.appendChild(imgElement);
  predictionContainer.appendChild(imgContainer);

  const probsContainer = document.createElement('div');
  for (let i = 0; i < classes.length; i++) {
    const row = document.createElement('div');
    row.className = 'row';

    const classElement = document.createElement('div');
    classElement.className = 'cell';
    classElement.innerText = classes[i].className;
    row.appendChild(classElement);

    const probsElement = document.createElement('div');
    probsElement.className = 'cell';
    probsElement.innerText = classes[i].probability.toFixed(0) + '%';
    row.appendChild(probsElement);

    probsContainer.appendChild(row);
  }
  predictionContainer.appendChild(probsContainer);

  predictionsElement.insertBefore(
      predictionContainer, predictionsElement.firstChild);
}

const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    const idx = i;
    // Closure to capture the file information.
    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      img.onload = () => predict(img);
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});

const predictionsElement = document.getElementById('predictions');

demo();