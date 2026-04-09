# ArchTakeoff — CV Measurement Engine

Automated architectural plan takeoff using Claude's vision API. Upload a floor plan,
site plan, or elevation drawing and get back annotated measurements with room-by-room
dimensions, area calculations, and exportable takeoff data.

---

## Quick Start (Local / Development)

1. Open `index.html` in a browser — **no build step required**.
2. Set your Anthropic API key in `js/config.js`:
   ```js
   ANTHROPIC_API_KEY: 'sk-ant-...',
   ```
3. Upload a plan image and click **Run Analysis**.

> ⚠️ Embedding your API key in client-side JS is fine for local testing but **not safe
> for production**. See the Proxy Server section below.

---

## Project Structure

```
arch-takeoff/
├── index.html          # Main app shell & HTML
├── css/
│   └── styles.css      # All styles (blueprint dark theme)
├── js/
│   ├── config.js       # API key & endpoint configuration  ← edit this
│   ├── state.js        # Shared app state & color constants
│   ├── ui.js           # Step navigation, logging, controls
│   ├── upload.js       # Drag-and-drop & file input
│   ├── analysis.js     # Claude API call & results rendering
│   ├── canvas.js       # Canvas overlay drawing
│   └── export.js       # CSV & PNG export
└── README.md
```

---

## Production: Proxy Server Setup

Never expose an API key in client-side code in production. Instead, route requests
through a lightweight server that injects the key server-side.

### Option A — Express proxy (Node.js)

1. Install dependencies:
   ```bash
   npm install express cors node-fetch
   ```

2. Create `server/proxy.js`:
   ```js
   const express = require('express');
   const cors    = require('cors');
   const fetch   = require('node-fetch');

   const app = express();
   app.use(cors());
   app.use(express.json({ limit: '20mb' }));

   app.post('/api/analyze', async (req, res) => {
     try {
       const response = await fetch('https://api.anthropic.com/v1/messages', {
         method:  'POST',
         headers: {
           'Content-Type':      'application/json',
           'x-api-key':         process.env.ANTHROPIC_API_KEY,
           'anthropic-version': '2023-06-01',
         },
         body: JSON.stringify(req.body),
       });
       const data = await response.json();
       res.json(data);
     } catch (err) {
       res.status(500).json({ error: { message: err.message } });
     }
   });

   app.listen(3001, () => console.log('Proxy running on http://localhost:3001'));
   ```

3. Run the server:
   ```bash
   ANTHROPIC_API_KEY=sk-ant-... node server/proxy.js
   ```

4. In `js/config.js`, update the endpoint and clear the key:
   ```js
   ANTHROPIC_API_KEY: null,
   API_ENDPOINT: 'http://localhost:3001/api/analyze',
   ```

### Option B — Serverless (Vercel / Netlify)

Create an API route (e.g. `api/analyze.js` for Vercel) that does the same fetch
as the Express proxy above, reading `process.env.ANTHROPIC_API_KEY` from your
deployment environment variables.

---

## Features

| Feature | Detail |
|---|---|
| **Scale detection** | Auto-reads scale bars, dimension strings, and title block annotations |
| **Manual override** | Enter your own scale (e.g. 1:50, 1:100, 1/4"=1') |
| **Room detection** | Bounding-box overlay per room with color coding |
| **Dimension lines** | Overall building extents with amber annotation lines |
| **Layer toggles** | Show/hide Rooms, Dims, and Labels independently |
| **Room highlight** | Click any room in the list to highlight it on the plan |
| **Export CSV** | Full takeoff spreadsheet — room names, dims, areas |
| **Export image** | Annotated plan as a PNG at the original image resolution |
| **Units** | Switch between metric (m/m²) and imperial (ft/ft²) |
| **Plan types** | Floor plan, site plan, elevation, section |
| **Detail levels** | Standard / Detailed (doors & windows) / Full |

---

## Tips for Best Results

- Use **clear, high-resolution** scanned plans or CAD exports (300 dpi+)
- Plans with a **visible scale bar** yield the highest confidence measurements
- **Title blocks** with a noted scale also work well for auto-detection
- Avoid heavily overlaid or coloured plans — cleaner line work = better results
- For hand-drawn sketches, use **manual scale** mode

---

## Model

Uses `claude-sonnet-4-20250514` by default. You can swap to `claude-opus-4-20250514`
in `js/config.js` for higher accuracy on complex plans (higher cost).

---

## License

MIT
