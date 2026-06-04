/**
 * Cloudflare Pages Function — proxy /api/cv-analyze to the Render Python service.
 *
 * Required Cloudflare Pages environment variables:
 *   CV_SERVICE_URL      — Base URL of the Render service, e.g. https://arqen-cv.onrender.com
 *   CV_SERVICE_SECRET   — Must match SERVICE_SECRET set in the Render environment
 *                         (copy the auto-generated value from the Render dashboard).
 *                         Leave blank if SERVICE_SECRET is not set on Render.
 */
export async function onRequestPost({ request, env }) {
  const cvUrl = (env.CV_SERVICE_URL || '').replace(/\/$/, '');
  if (!cvUrl) {
    return Response.json(
      { error: 'CV_SERVICE_URL is not configured in Cloudflare Pages environment variables.' },
      { status: 503 }
    );
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: 'Invalid JSON body.' }, { status: 400 });
  }

  const headers = { 'Content-Type': 'application/json' };
  const secret = env.CV_SERVICE_SECRET || '';
  if (secret) headers['X-Service-Secret'] = secret;

  try {
    const upstream = await fetch(`${cvUrl}/cv-analyze`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });
    const data = await upstream.json();
    return Response.json(data, { status: upstream.status });
  } catch (err) {
    return Response.json(
      { error: 'CV service unreachable: ' + err.message },
      { status: 502 }
    );
  }
}
