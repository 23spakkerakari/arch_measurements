/**
 * Cloudflare Worker entry — route /api/* to Pages-style handlers, serve static UI otherwise.
 */
import { onRequestPost as cvAnalyze } from './functions/api/cv-analyze.js';
import { onRequestPost as analyze } from './functions/api/analyze.js';

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (url.pathname === '/api/cv-analyze' && request.method === 'POST') {
      return cvAnalyze({ request, env });
    }
    if (url.pathname === '/api/analyze' && request.method === 'POST') {
      return analyze({ request, env });
    }

    return env.ASSETS.fetch(request);
  },
};
