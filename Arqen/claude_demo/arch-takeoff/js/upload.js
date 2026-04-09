/**
 * upload.js
 * Handles file drag-and-drop and file input selection for images and PDFs.
 */

const ACCEPTED_IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/gif', 'image/webp'];
const ACCEPTED_PDF_TYPE    = 'application/pdf';

const dropZone  = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

pdfjsLib.GlobalWorkerOptions.workerSrc =
  'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragging');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragging');
});

dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragging');
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});

fileInput.addEventListener('change', e => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

function handleFile(f) {
  const isImage = ACCEPTED_IMAGE_TYPES.includes(f.type) || f.type.startsWith('image/');
  const isPdf   = f.type === ACCEPTED_PDF_TYPE || f.name.toLowerCase().endsWith('.pdf');

  if (!isImage && !isPdf) {
    alert('Please upload an image (PNG, JPG, GIF, WEBP) or a PDF file.');
    return;
  }

  appState.imageFile = f;
  appState.fileType  = isPdf ? 'pdf' : 'image';

  const reader = new FileReader();
  reader.onload = ev => {
    appState.fileDataUrl = ev.target.result;

    if (isPdf) {
      renderPdfFirstPage(ev.target.result);
    } else {
      appState.imageDataUrl = ev.target.result;
      document.getElementById('preview-img').src = ev.target.result;
      document.getElementById('result-img').src  = ev.target.result;
      goToStep(2);
    }
  };
  reader.readAsDataURL(f);
}

/**
 * Renders page 1 of a PDF to an off-screen canvas, converts it to a data URL,
 * and uses that as the preview / result image so the rest of the app works unchanged.
 */
async function renderPdfFirstPage(dataUrl) {
  try {
    const raw    = atob(dataUrl.split(',')[1]);
    const bytes  = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);

    const pdf  = await pdfjsLib.getDocument({ data: bytes }).promise;
    const page = await pdf.getPage(1);

    const scale    = 2;
    const viewport = page.getViewport({ scale });
    const canvas   = document.createElement('canvas');
    canvas.width   = viewport.width;
    canvas.height  = viewport.height;

    await page.render({ canvasContext: canvas.getContext('2d'), viewport }).promise;

    const imageUrl = canvas.toDataURL('image/png');
    appState.imageDataUrl = imageUrl;
    document.getElementById('preview-img').src = imageUrl;
    document.getElementById('result-img').src  = imageUrl;

    const pageCount = pdf.numPages;
    if (pageCount > 1) {
      console.log(`PDF has ${pageCount} pages — showing page 1 as preview; all pages sent to API.`);
    }

    goToStep(2);
  } catch (err) {
    console.error('PDF render error:', err);
    alert('Could not render PDF preview. The file may be corrupted or password-protected.');
  }
}
