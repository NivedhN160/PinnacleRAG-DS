'use client';
import { useSearchParams, useRouter } from 'next/navigation';
import { useState, useRef, useEffect, Suspense } from 'react';
import ParticleCanvas from '@/components/ParticleCanvas';
import Link from 'next/link';

interface Citation {
  id: number;
  source: string;
  snippet: string;
}

interface Message {
  role: string;
  content: string;
  citations?: Citation[];
  rewritten_query?: string;
}

function ChatInterface() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const domain = searchParams.get('domain') || 'general';
  
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  
  // Feature toggles
  const [rewrite, setRewrite] = useState(false);
  const [checkFaithfulness, setCheckFaithfulness] = useState(false);
  const [budgetRemaining, setBudgetRemaining] = useState<number | null>(null);
  
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
  
  // Upload panel state
  const [showUpload, setShowUpload] = useState(false);
  const [uploadMode, setUploadMode] = useState<'file' | 'text'>('file');
  const [files, setFiles] = useState<FileList | null>(null);
  const [pasteText, setPasteText] = useState('');
  const [pasteFilename, setPasteFilename] = useState('');
  const [uploadLoading, setUploadLoading] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch(`${apiUrl}/api/query/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          question: userMessage.content, 
          mode: 'agent', 
          domain,
          rewrite: rewrite,
          check_faithfulness: checkFaithfulness
        })
      });
      
      if (!res.ok) throw new Error('Failed to fetch response');
      
      const data = await res.json();
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: data.answer, 
        citations: data.citations,
        rewritten_query: data.rewritten_query
      }]);
      if (data.usage?.budget_remaining_calls !== undefined) {
        setBudgetRemaining(data.usage.budget_remaining_calls);
      }
    } catch (error) {
      setMessages(prev => [...prev, { role: 'system', content: 'Error communicating with the backend. Is uvicorn running on port 8000?' }]);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    setUploadLoading(true);
    
    try {
      let res;
      let data;
      if (uploadMode === 'file' && files && files.length > 0) {
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
          formData.append('files', files[i]);
        }
        formData.append('domain', domain);
        formData.append('rebuild', 'true');
        
        res = await fetch(`${apiUrl}/api/ingest/upload`, {
          method: 'POST',
          body: formData,
        });
      } else if (uploadMode === 'text' && pasteText && pasteFilename) {
        res = await fetch(`${apiUrl}/api/ingest/text`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: pasteText,
            filename: pasteFilename,
            domain: domain,
            rebuild: true
          }),
        });
      }
      
      if (res && !res.ok) throw new Error('Upload failed');
      if (res) data = await res.json();
      
      const stats = data?.ingest_stats;
      const statsMsg = stats ? ` (Loaded ${stats.documents_loaded} docs into ${stats.chunks_created} chunks)` : '';
      setMessages(prev => [...prev, { role: 'system', content: `Successfully indexed data into the ${domain} domain.${statsMsg}` }]);
      setShowUpload(false);
      setFiles(null);
      setPasteText('');
      setPasteFilename('');
    } catch (error) {
      setMessages(prev => [...prev, { role: 'system', content: 'Error uploading data.' }]);
    } finally {
      setUploadLoading(false);
    }
  };

  const domainColors: Record<string, string> = {
    trading: 'text-blue-400 bg-blue-400/10 border-blue-400/30',
    security: 'text-red-400 bg-red-400/10 border-red-400/30',
    seo: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30',
    general: 'text-slate-400 bg-slate-400/10 border-slate-400/30'
  };
  
  const domains = ['general', 'trading', 'security', 'seo'];

  return (
    <div className="relative min-h-screen text-slate-50 font-sans flex flex-col">
      <ParticleCanvas />
      
      {/* Header */}
      <header className="relative z-10 p-6 border-b border-slate-800 bg-slate-900/50 backdrop-blur-md flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-pink-400">PinnacleRAG-DS</h1>
          <div className="flex gap-2 mt-2">
            {domains.map(d => (
              <button
                key={d}
                onClick={() => router.push(`/chat?domain=${d}`)}
                className={`px-3 py-1 rounded-full text-xs font-semibold capitalize border transition-all ${
                  domain === d ? domainColors[d] : 'border-slate-700 text-slate-500 hover:text-slate-300'
                }`}
              >
                {d}
              </button>
            ))}
          </div>
        </div>
        <div className="flex gap-4 items-center">
          {budgetRemaining !== null && (
            <span className="text-xs font-semibold text-slate-400 bg-slate-800 px-2 py-1 rounded">
              Budget: {budgetRemaining} left
            </span>
          )}
          <button 
            onClick={() => setShowUpload(!showUpload)}
            className="px-4 py-2 rounded-lg bg-indigo-600/20 text-indigo-400 hover:bg-indigo-600/30 transition-colors text-sm font-medium border border-indigo-500/30"
          >
            {showUpload ? 'Close Upload' : 'Upload Data'}
          </button>
          <Link href="/" className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition-colors text-sm font-medium border border-slate-700">
            ← Back
          </Link>
        </div>
      </header>
      
      {/* Upload Panel */}
      {showUpload && (
        <div className="relative z-20 bg-slate-800/95 backdrop-blur-xl border-b border-slate-700 p-6">
          <div className="max-w-4xl mx-auto">
            <div className="flex gap-4 mb-4">
              <button 
                onClick={() => setUploadMode('file')}
                className={`px-4 py-2 rounded-lg text-sm font-medium ${uploadMode === 'file' ? 'bg-indigo-600 text-white' : 'bg-slate-700 text-slate-300'}`}
              >
                File Upload
              </button>
              <button 
                onClick={() => setUploadMode('text')}
                className={`px-4 py-2 rounded-lg text-sm font-medium ${uploadMode === 'text' ? 'bg-indigo-600 text-white' : 'bg-slate-700 text-slate-300'}`}
              >
                Paste Text
              </button>
            </div>
            
            <form onSubmit={handleUpload} className="flex flex-col gap-4">
              {uploadMode === 'file' ? (
                <input 
                  type="file" 
                  multiple
                  onChange={e => setFiles(e.target.files)}
                  className="bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-indigo-600 file:text-white"
                />
              ) : (
                <>
                  <input 
                    type="text"
                    placeholder="Filename (e.g. notes.txt)"
                    value={pasteFilename}
                    onChange={e => setPasteFilename(e.target.value)}
                    className="bg-slate-900 border border-slate-700 rounded-lg p-3 text-slate-300"
                  />
                  <textarea 
                    placeholder="Paste text here..."
                    value={pasteText}
                    onChange={e => setPasteText(e.target.value)}
                    className="bg-slate-900 border border-slate-700 rounded-lg p-3 text-slate-300 min-h-[120px]"
                  />
                </>
              )}
              
              <button 
                type="submit"
                disabled={uploadLoading || (uploadMode === 'file' ? (!files || files.length === 0) : (!pasteText || !pasteFilename))}
                className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white px-6 py-3 rounded-xl font-medium self-end"
              >
                {uploadLoading ? 'Indexing...' : `Upload to ${domain} Domain`}
              </button>
            </form>
          </div>
        </div>
      )}

      {/* Chat Area */}
      <main className="relative z-10 flex-1 overflow-y-auto p-4 md:p-8 flex flex-col gap-4 max-w-4xl mx-auto w-full">
        {messages.length === 0 && (
          <div className="flex-1 flex flex-col items-center justify-center text-center opacity-50">
            <h2 className="text-3xl font-semibold mb-2">Agent is Ready</h2>
            <p className="max-w-md mb-6">Ask a question to test the {domain} domain specialization. It will ground its answers on documents retrieved from your local index.</p>
            <div className="flex flex-wrap justify-center gap-2 max-w-lg">
              {domain === 'trading' && [
                "What is the PE ratio of Apple?",
                "Analyze the recent FED rate hike impact on tech stocks."
              ].map(q => (
                <button key={q} onClick={() => setInput(q)} className="text-xs bg-slate-800 hover:bg-slate-700 px-3 py-2 rounded-full transition-colors border border-slate-700">{q}</button>
              ))}
              {domain === 'security' && [
                "What are the mitigation steps for Log4Shell?",
                "Explain the zero trust architecture."
              ].map(q => (
                <button key={q} onClick={() => setInput(q)} className="text-xs bg-slate-800 hover:bg-slate-700 px-3 py-2 rounded-full transition-colors border border-slate-700">{q}</button>
              ))}
              {domain === 'seo' && [
                "How can I improve my core web vitals?",
                "What is the impact of canonical tags?"
              ].map(q => (
                <button key={q} onClick={() => setInput(q)} className="text-xs bg-slate-800 hover:bg-slate-700 px-3 py-2 rounded-full transition-colors border border-slate-700">{q}</button>
              ))}
              {domain === 'general' && [
                "Summarize the main points of the uploaded document.",
                "What are the key takeaways?"
              ].map(q => (
                <button key={q} onClick={() => setInput(q)} className="text-xs bg-slate-800 hover:bg-slate-700 px-3 py-2 rounded-full transition-colors border border-slate-700">{q}</button>
              ))}
            </div>
          </div>
        )}
        
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] rounded-2xl px-6 py-4 ${
              msg.role === 'user' 
                ? 'bg-indigo-600/80 backdrop-blur-sm text-white rounded-tr-sm' 
                : msg.role === 'system'
                  ? 'bg-red-500/20 text-red-200 border border-red-500/30'
                  : 'bg-slate-800/80 backdrop-blur-sm border border-slate-700 text-slate-200 rounded-tl-sm'
            }`}>
              {msg.rewritten_query && (
                <div className="text-xs text-indigo-300 mb-2 font-medium italic">
                  Rewritten Query: {msg.rewritten_query}
                </div>
              )}
              <div className="whitespace-pre-wrap">{msg.content}</div>
              
              {/* Citations block */}
              {msg.citations && msg.citations.length > 0 && (
                <div className="mt-4 pt-4 border-t border-slate-700/50">
                  <p className="text-xs font-semibold text-slate-400 mb-2">Sources:</p>
                  <div className="flex flex-col gap-2">
                    {msg.citations.map((c, cIdx) => (
                      <div key={cIdx} className="text-xs bg-slate-900/50 p-2 rounded border border-slate-700/50">
                        <span className="text-indigo-400 font-mono mr-2">[{c.id}]</span>
                        <span className="text-slate-300 break-all">{c.source}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="max-w-[80%] rounded-2xl px-6 py-4 bg-slate-800/80 backdrop-blur-sm border border-slate-700 text-slate-400 rounded-tl-sm flex gap-2 items-center">
              <span className="w-2 h-2 bg-slate-500 rounded-full animate-bounce"></span>
              <span className="w-2 h-2 bg-slate-500 rounded-full animate-bounce" style={{animationDelay: '150ms'}}></span>
              <span className="w-2 h-2 bg-slate-500 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </main>

      <footer className="relative z-10 p-4 bg-slate-900/80 backdrop-blur-lg border-t border-slate-800 flex flex-col items-center">
        <div className="max-w-4xl w-full flex justify-end gap-4 mb-3 px-2">
          <label className="flex items-center gap-2 text-xs text-slate-400 cursor-pointer hover:text-slate-200 transition-colors">
            <input 
              type="checkbox" 
              checked={rewrite} 
              onChange={e => setRewrite(e.target.checked)}
              className="rounded bg-slate-800 border-slate-700 text-indigo-600 focus:ring-indigo-500/50"
            />
            Rewrite Query
          </label>
          <label className="flex items-center gap-2 text-xs text-slate-400 cursor-pointer hover:text-slate-200 transition-colors">
            <input 
              type="checkbox" 
              checked={checkFaithfulness} 
              onChange={e => setCheckFaithfulness(e.target.checked)}
              className="rounded bg-slate-800 border-slate-700 text-indigo-600 focus:ring-indigo-500/50"
            />
            Check Faithfulness
          </label>
        </div>
        <form onSubmit={handleSubmit} className="max-w-4xl w-full flex gap-3">
          <input 
            type="text" 
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder={`Ask the ${domain} agent...`}
            className="flex-1 bg-slate-800/50 border border-slate-700 rounded-xl px-6 py-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all placeholder:text-slate-500"
            disabled={loading}
          />
          <button 
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white px-8 py-4 rounded-xl font-medium transition-all shadow-lg shadow-indigo-500/20"
          >
            Send
          </button>
        </form>
      </footer>
    </div>
  );
}

export default function ChatPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-slate-900 text-white flex items-center justify-center">Loading...</div>}>
      <ChatInterface />
    </Suspense>
  );
}
