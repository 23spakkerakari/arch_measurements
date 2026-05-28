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

  // API endpoint — relative so it works both locally (Express on :3001) and on Cloudflare Pages
  API_ENDPOINT: '/api/analyze',

  MODEL: 'claude-sonnet-4-6',
  MAX_TOKENS: 16384,
};
