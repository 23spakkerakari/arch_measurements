/**
 * config.js
 *
 * Central configuration for ArchTakeoff.
 *
 * SETUP: Set your Anthropic API key below, or supply it via an
 * environment variable if running through a local proxy server.
 *
 * ⚠️  For production use, never expose your API key in client-side code.
 *     Route requests through your own backend server instead.
 *     See README.md for the recommended server-side proxy setup.
 */

const CONFIG = {
  // ------------------------------------------------------------------
  // Replace with your Anthropic API key for direct browser testing.
  // For production, set this to null and use the proxy server in
  // server/proxy.js — see README.md for instructions.
  // ------------------------------------------------------------------
  ANTHROPIC_API_KEY: null,

  // API endpoint — routed through the local proxy to avoid CORS
  API_ENDPOINT: 'http://localhost:3001/api/analyze',

  MODEL: 'claude-sonnet-4-6',
  MAX_TOKENS: 8192,
};
