// frontend/src/utils/renderAnswer.jsx
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

/**
 * Turn bare "[n]" citations into Markdown links "[n](url)".
 * If a source for that number doesn't exist, leave the text unchanged.
 */
function linkifyCitations(md, sources) {
  return md.replace(/\[(\d+)]/g, (match, n) => {
    const src = sources[Number(n) - 1];
    return src ? `[${n}](${src.url})` : match;
  });
}

export default function RenderAnswer({ text, sources }) {
  const mdWithLinks = linkifyCitations(text, sources);

  return (
    <div className="prose prose-slate max-w-none leading-relaxed break-words">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          /**
           * Render any link whose visible text is just a number
           * (our inline citation) as a small round "chip".
           * All other links fall back to a normal underlined style.
           */
          a({ href, children }) {
            const label =
              typeof children?.[0] === 'string' ? children[0] : '';

            if (/^\d+$/.test(label)) {
              return (
                <a
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="
                    inline-flex items-center justify-center
                    mx-1 h-6 w-6 text-xs font-semibold
                    rounded-full bg-blue-100 text-blue-700
                    hover:bg-blue-200 transition-all duration-200
                    border border-blue-200 hover:border-blue-300
                    shadow-sm hover:shadow-md
                  "
                  title={href}
                >
                  {label}
                </a>
              );
            }

            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:text-blue-700 underline decoration-blue-300 hover:decoration-blue-400 transition-colors duration-200"
              >
                {children}
              </a>
            );
          },
          p({ children }) {
            return (
              <p className="text-slate-700 leading-relaxed mb-4 last:mb-0">
                {children}
              </p>
            );
          },
          h1({ children }) {
            return (
              <h1 className="text-2xl font-bold text-slate-900 mb-4 mt-6 first:mt-0">
                {children}
              </h1>
            );
          },
          h2({ children }) {
            return (
              <h2 className="text-xl font-semibold text-slate-900 mb-3 mt-5 first:mt-0">
                {children}
              </h2>
            );
          },
          h3({ children }) {
            return (
              <h3 className="text-lg font-semibold text-slate-900 mb-2 mt-4 first:mt-0">
                {children}
              </h3>
            );
          },
          ul({ children }) {
            return (
              <ul className="list-disc list-inside text-slate-700 mb-4 space-y-1">
                {children}
              </ul>
            );
          },
          ol({ children }) {
            return (
              <ol className="list-decimal list-inside text-slate-700 mb-4 space-y-1">
                {children}
              </ol>
            );
          },
          li({ children }) {
            return (
              <li className="text-slate-700 leading-relaxed">
                {children}
              </li>
            );
          },
          blockquote({ children }) {
            return (
              <blockquote className="border-l-4 border-blue-200 pl-4 py-2 my-4 bg-blue-50 rounded-r-lg">
                <div className="text-slate-700 italic">
                  {children}
                </div>
              </blockquote>
            );
          },
          code({ children, className }) {
            if (className) {
              // Code block
              return (
                <pre className="bg-slate-100 border border-slate-200 rounded-lg p-4 overflow-x-auto my-4">
                  <code className="text-sm text-slate-800 font-mono">
                    {children}
                  </code>
                </pre>
              );
            }
            // Inline code
            return (
              <code className="bg-slate-100 text-slate-800 px-1.5 py-0.5 rounded text-sm font-mono">
                {children}
              </code>
            );
          },
          strong({ children }) {
            return (
              <strong className="font-semibold text-slate-900">
                {children}
              </strong>
            );
          },
          em({ children }) {
            return (
              <em className="italic text-slate-700">
                {children}
              </em>
            );
          },
        }}
      >
        {mdWithLinks}
      </ReactMarkdown>
    </div>
  );
}
