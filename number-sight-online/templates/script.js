const canvas = document.getElementById('image-area');
const ctx = canvas.getContext('2d');

function blankPixelArray() {
  const pixels = []
  for (let h = 0; h < height; h++) {
    let row = []
    for (let w = 0; w < width; w++) {
      row.push(0)
    }
    pixels.push(row)
  }

  return pixels
}

function clearCanvas() {
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  pixels = blankPixelArray();
}

function saveImage() {
  let out = "";
  pixels.forEach((row, y) => {
    row.forEach((value, x) => {
      out += value;
    });
  });
  out += document.getElementById('answer-input').value;
  fetch('/savetrainingdata', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ trainingData: out }),
  }).then(() => {
    clearCanvas();
  }).catch((error) => {
    alert('Error saving image: ' + error.message)
  })
}


function onDrag(event) {
  if (!mouseDown) return;
  const coords = getCoords(event);
  ctx.fillStyle = 'white'; // Or any desired drawing color
  ctx.fillRect(coords.x * pixelSize, coords.y * pixelSize, pixelSize, pixelSize);
  pixels[coords.y][coords.x] = 1;
}

function getCoords(event) {
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor((event.clientX - rect.left) / pixelSize);
  const y = Math.floor((event.clientY - rect.top) / pixelSize);
  return { x, y };
}

function mouseUp() {
  mouseDown = false;
  fetch('/getprediction', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ pixels }),
  })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById('prediction').innerText = data.number + ' with ' + data.confidence + '% confidence';
    });

}

const pixelSize = 10;
const width = 20;
const height = 20;
let pixels = blankPixelArray();

canvas.width = width * pixelSize;
canvas.height = height * pixelSize;

let mouseDown = false;

clearCanvas();
canvas.addEventListener('mousedown', () => { mouseDown = true; });
canvas.addEventListener('mouseup', mouseUp);
canvas.addEventListener('mousemove', onDrag);



