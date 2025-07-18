// frontend/src/api.js
const BASE = 'http://127.0.0.1:8000';

/**
 * Stream answer chunks (newline‑delimited JSON) from the backend.
 * Returns an async iterator that yields events:
 *   {type:'sources', data:[{title,url}…]}
 *   {type:'token',   value:'string'}
 *   {type:'end',     answer:'full answer', sources:[…]}
 *   {type:'training_start'} - Training mode started
 *   {type:'token',   value:'string', response:1|2} - Training mode tokens
 *   {type:'training_comparison', response1:{...}, response2:{...}} - Training comparison
 *   {type:'training_end'} - Training mode ended
 */
export async function askStream(query, mode = 'vanilla', model = 'gpt4o', count = 5, training = false) {
  const params = new URLSearchParams({
    q: query,
    mode,
    model,
    count: String(count),
    stream: 'true',
    training: String(training),
  });
  const res = await fetch(`${BASE}/answer?${params.toString()}`);
  if (!res.ok) throw new Error(`Backend error ${res.status}`);

  const reader   = res.body.getReader();
  const decoder  = new TextDecoder('utf-8');
  let buffer     = '';

  async function* iterator() {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, idx).trim();
        buffer     = buffer.slice(idx + 1);
        if (line) yield JSON.parse(line);
      }
    }
  }
  return iterator();
}
