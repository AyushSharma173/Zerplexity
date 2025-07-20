// frontend/src/api.js
// API configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export async function askStream(q, mode, model, count, training, queryGenerator) {
  const params = new URLSearchParams({
    q,
    count,
    mode,
    model,
    training: training.toString(),
    stream: 'true',
    query_generator: queryGenerator
  });

  const response = await fetch(`${API_BASE_URL}/answer?${params}`);
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  return {
    async *[Symbol.asyncIterator]() {
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n').filter(line => line.trim());

          for (const line of lines) {
            try {
              const data = JSON.parse(line);
              yield data;
            } catch (e) {
              console.warn('Failed to parse JSON:', line);
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    }
  };
}

export async function searchWeb(q, count = 10) {
  const params = new URLSearchParams({ q, count });
  const response = await fetch(`${API_BASE_URL}/search?${params}`);
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return response.json();
}

export async function generateQueries(q, queryGenerator = 'gpt4o') {
  const params = new URLSearchParams({ q, query_generator: queryGenerator });
  const response = await fetch(`${API_BASE_URL}/generate_queries?${params}`);
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return response.json();
}
