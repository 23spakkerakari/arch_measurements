const express = require('express');
const cors    = require('cors');
const path    = require('path');
const { spawn } = require('child_process');
const fs = require('fs').promises;
const os = require('os');

const API_KEY = process.env.ANTHROPIC_API_KEY || null;
const PORT    = 3001;

const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));

app.use(express.static(path.join(__dirname, '..')));

app.post('/api/cv-analyze', async (req, res) => {
  const { imageBase64, scale, dpi = 150, roi } = req.body || {};
  if (!imageBase64 || !scale) {
    return res.status(400).json({ error: 'imageBase64 and scale are required' });
  }

  let tmpDir;
  try {
    tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'arqen-cv-'));
    const imgPath = path.join(tmpDir, 'plan.png');
    const b64 = imageBase64.replace(/^data:image\/\w+;base64,/, '');
    await fs.writeFile(imgPath, Buffer.from(b64, 'base64'));

    const scriptPath = path.join(__dirname, '..', 'scripts', 'cv_analyze.py');
    const arqenRoot = path.join(__dirname, '..', '..', '..');

    const pyArgs = [scriptPath, '--image', imgPath, '--scale', scale, '--dpi', String(dpi)];
    if (roi && roi.x0_pct != null) {
      pyArgs.push(
        '--roi',
        `${roi.x0_pct},${roi.y0_pct},${roi.x1_pct},${roi.y1_pct}`,
      );
    }

    const stdout = await new Promise((resolve, reject) => {
      const py = spawn(
        'python',
        pyArgs,
        { cwd: arqenRoot, env: { ...process.env, PYTHONPATH: arqenRoot } },
      );
      let out = '';
      let err = '';
      py.stdout.on('data', chunk => { out += chunk; });
      py.stderr.on('data', chunk => { err += chunk; console.error('[cv]', chunk.toString()); });
      const timer = setTimeout(() => {
        py.kill();
        reject(new Error('CV analysis timed out after 120s'));
      }, 120000);
      py.on('error', e => { clearTimeout(timer); reject(e); });
      py.on('close', code => {
        clearTimeout(timer);
        if (code !== 0) reject(new Error(err || `CV exited with code ${code}`));
        else resolve(out);
      });
    });

    const result = JSON.parse(stdout);
    res.json(result);
  } catch (err) {
    console.error('CV analyze error:', err.message);
    res.status(500).json({ error: err.message });
  } finally {
    if (tmpDir) await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {});
  }
});

app.post('/api/analyze', async (req, res) => {
  const key = API_KEY || req.headers['x-api-key'];
  if (!key) {
    return res.status(401).json({ error: { message: 'No API key configured. Set ANTHROPIC_API_KEY env var.' } });
  }

  const headers = {
    'Content-Type':      'application/json',
    'x-api-key':         key,
    'anthropic-version': '2023-06-01',
  };

  const beta = req.headers['anthropic-beta'];
  if (beta) headers['anthropic-beta'] = beta;

  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method:  'POST',
      headers,
      body:    JSON.stringify(req.body),
    });

    const data = await response.json();

    if (!response.ok) {
      return res.status(response.status).json(data);
    }

    res.json(data);
  } catch (err) {
    console.error('Proxy error:', err.message);
    res.status(502).json({ error: { message: 'Proxy failed to reach Anthropic API: ' + err.message } });
  }
});

app.listen(PORT, () => {
  console.log(`ArchTakeoff proxy running at http://localhost:${PORT}`);
  console.log(`Open http://localhost:${PORT}/index.html in your browser`);
  console.log(API_KEY ? 'API key: loaded from env' : 'API key: will be read from client request header');
});
