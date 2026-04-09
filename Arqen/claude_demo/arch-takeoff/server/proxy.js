const express = require('express');
const cors    = require('cors');
const path    = require('path');

const API_KEY = process.env.ANTHROPIC_API_KEY || null;
const PORT    = 3001;

const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));

app.use(express.static(path.join(__dirname, '..')));

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
