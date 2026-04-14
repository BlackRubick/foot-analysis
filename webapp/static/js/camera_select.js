// camera_select.js
// Permite seleccionar la cámara antes de abrir el stream en todos los módulos

let availableCameras = [];
let currentStream = null;

async function getCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter(device => device.kind === 'videoinput');
}

async function populateCameraSelect(selectId) {
  const cameras = await getCameras();
  availableCameras = cameras;
  const select = document.getElementById(selectId);
  select.innerHTML = '';
  cameras.forEach((cam, idx) => {
    const option = document.createElement('option');
    option.value = cam.deviceId;
    option.text = cam.label || `Cámara ${idx+1}`;
    select.appendChild(option);
  });
}

async function openCamera(module) {
  await populateCameraSelect(`${module}_camera_select`);
  document.getElementById(`${module}-camera`).classList.remove('hidden');
  const select = document.getElementById(`${module}_camera_select`);
  select.onchange = () => startCamera(module);
  startCamera(module);
}

async function startCamera(module) {
  const select = document.getElementById(`${module}_camera_select`);
  const deviceId = select.value;
  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
  }
  const constraints = {
    video: { deviceId: { exact: deviceId } }
  };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  currentStream = stream;
  document.getElementById(`${module}_video`).srcObject = stream;
}

function closeCamera(module) {
  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
    currentStream = null;
  }
  document.getElementById(`${module}-camera`).classList.add('hidden');
}

function captureFromCamera(module) {
  const video = document.getElementById(`${module}_video`);
  const canvas = document.getElementById(`${module}_canvas`);
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  canvas.classList.remove('hidden');
  // Guardar la imagen capturada en el input hidden correspondiente
  let inputId = `${module}_data`;
  // Para los módulos de rodilla y postura, el input hidden tiene nombre diferente
  if (module === 'knee_frontal') inputId = 'knee_image_frontal_data';
  if (module === 'knee_sagital') inputId = 'knee_image_sagital_data';
  if (module === 'posture_frontal') inputId = 'posture_image_frontal_data';
  if (module === 'posture_sagital') inputId = 'posture_image_sagital_data';
  const input = document.getElementById(inputId);
  if (input) {
    input.value = canvas.toDataURL('image/png');
  }
}
