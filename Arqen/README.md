# ArchTakeoff — Wall Measurement Engine

Extracts exterior wall measurements, facing directions, and total area from
architectural plan drawings using Claude's vision API. Upload a floor plan
(image or PDF), configure the scale, and get structured JSON takeoff data.

## Quick Start

### 1. Install dependencies

```bash
cd claude_demo/arch-takeoff
npm install
```

### 2. Start the proxy server

The app routes requests through a local Express proxy to avoid browser CORS
restrictions. Pass your Anthropic API key as an environment variable:

**PowerShell:**
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-..."
node server/proxy.js
```

**Bash / macOS / Linux:**
```bash
ANTHROPIC_API_KEY=sk-ant-... node server/proxy.js
```

You should see:
```
ArchTakeoff proxy running at http://localhost:3001
Open http://localhost:3001/index.html in your browser
API key: loaded from env
```

### 3. Open the app

Go to **http://localhost:3001/index.html** in your browser.

## Usage

The app follows a 4-step workflow:

### Step 1 — Upload Plan

Drag and drop (or browse for) an architectural plan. Supported formats:

- **Images:** PNG, JPG, GIF, WEBP
- **PDFs:** Multi-page PDFs are supported — all pages are sent to the API;
  page 1 is shown as the preview.

### Step 2 — Configure Scale

| Setting | Options | Notes |
|---|---|---|
| **Scale detection** | Auto Detect / Manual | Manual forces the model to use your exact scale |
| **Drawing scale** | e.g. `1:100`, `1/4"=1'` | Only shown when Manual is selected |
| **Units** | Metric (m) / Imperial (ft) | Controls the output unit system |
| **Plan type** | Floor Plan, Site Plan, Elevation, Section, Auto-detect | Helps the model interpret the drawing |

### Step 3 — Analyze

Click **RUN ANALYSIS**. The app sends the plan to Claude Sonnet 4.6 and
streams progress updates while waiting (~10–30 seconds depending on file size).

### Step 4 — Results

The results panel shows:

- **Summary metrics:** walls detected, total area, detected scale, confidence
- **Wall measurements:** each wall with its name, facing direction (N/S/E/W),
  and length
- **Analyst notes:** any observations the model flagged

#### Detail levels

| Level | What it extracts |
|---|---|
| Standard | Main exterior walls and total area |
| Detailed | All walls including interior partitions |
| Full | Every wall with facing direction and notes |

#### Export

- **CSV** — wall name, facing, length, and notes in spreadsheet format
- **IMG** — the plan image with dimension line overlays as a PNG

## Project Structure

```
arch-takeoff/
├── index.html              # App shell
├── css/styles.css          # Blueprint dark theme
├── js/
│   ├── config.js           # API key, endpoint, model settings
│   ├── state.js            # Shared app state
│   ├── ui.js               # Step navigation, logging, controls
│   ├── upload.js           # Drag-and-drop, PDF rendering (PDF.js)
│   ├── analysis.js         # Claude API call, prompt, results rendering
│   ├── canvas.js           # Dimension line overlays
│   └── export.js           # CSV and PNG export
├── server/
│   └── proxy.js            # Express proxy (injects API key server-side)
├── test_pdfs.py            # Batch test script for PDFs in data/
└── package.json
```

## Configuration

All settings live in `js/config.js`:

```js
const CONFIG = {
  ANTHROPIC_API_KEY: null,    // null when using the proxy (recommended)
  API_ENDPOINT: 'http://localhost:3001/api/analyze',
  MODEL: 'claude-sonnet-4-6',
  MAX_TOKENS: 8192,
};
```

- **`ANTHROPIC_API_KEY`** — leave as `null` when using the proxy. The proxy
  reads the key from the `ANTHROPIC_API_KEY` environment variable.
- **`MODEL`** — `claude-sonnet-4-6` is the default. Switch to
  `claude-opus-4-20250514` for higher accuracy on complex plans (higher cost).
- **`MAX_TOKENS`** — increase if large plans produce truncated output.

## Batch Testing

`test_pdfs.py` runs the first 10 PDFs from `../../data/` through the API and
saves results:

```bash
set ANTHROPIC_API_KEY=sk-ant-...
python test_pdfs.py
```

Results are printed to stdout and saved to `test_results.json`.

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Failed to fetch` | Proxy server not running | Start it with `node server/proxy.js` |
| `CONFIG is not defined` | Syntax error in `config.js` | Check that the API key string has quotes |
| `Could not parse AI response as JSON` | Model returned malformed JSON | Try again, or increase `MAX_TOKENS` |
| `API error 401` | Invalid or missing API key | Check your `ANTHROPIC_API_KEY` env var |
| PDF preview is blank | PDF.js failed to render | Check browser console; file may be corrupted |

## License

MIT
