'use client';
import { useSearchParams } from 'next/navigation';
import { useState, useRef, useEffect, Suspense } from 'react';
import ParticleCanvas from '@/components/ParticleCanvas';
import Link from 'next/link';

function ChatInterface() {
  const searchParams = useSearchParams();
  const domain = searchParams.get('domain') || 'general';
  
  const [messages, setMessages] = useState<{role: string, content: string}[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
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
      const res = await fetch('http://127.0.0.1:8000/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMessage.content, mode: 'agent', domain })
      });
      
      if (!res.ok) throw new Error('Failed to fetch response');
      
      const data = await res.json();
      setMessages(prev => [...prev, { role: 'assistant', content: data.answer }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'system', content: 'Error communicating with the backend. Is uvicorn running on port 8000?' }]);
    } finally {
      setLoading(false);
    }
  };

  const domainColors: Record<string, string> = {
    trading: 'text-blue-400',
    security: 'text-red-400',
    seo: 'text-emerald-400',
    general: 'text-slate-400'
  };

  return (
    <div className="relative min-h-screen text-slate-50 font-sans flex flex-col">
      <ParticleCanvas />
      
      {/* Header */}
      <header className="relative z-10 p-6 border-b border-slate-800 bg-slate-900/50 backdrop-blur-md flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-pink-400">PinnacleRAG-DS</h1>
          <p className="text-sm text-slate-400 mt-1">
            Active Domain: <span className={`font-semibold capitalize ${domainColors[domain] || 'text-slate-400'}`}>{domain}</span>
          </p>
        </div>
        <Link href="/" className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition-colors text-sm font-medium border border-slate-700">
          ← Back to Hub
        </Link>
      </header>

      {/* Chat Area */}
      <main className="relative z-10 flex-1 overflow-y-auto p-4 md:p-8 flex flex-col gap-4 max-w-4xl mx-auto w-full">
        {messages.length === 0 && (
          <div className="flex-1 flex flex-col items-center justify-center text-center opacity-50">
            <h2 className="text-3xl font-semibold mb-2">Agent is Ready</h2>
            <p className="max-w-md">Ask a question to test the {domain} domain specialization. It will ground its answers on documents retrieved from your local index.</p>
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
              <div className="whitespace-pre-wrap">{msg.content}</div>
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

      {/* Input Area */}
      <footer className="relative z-10 p-4 bg-slate-900/80 backdrop-blur-lg border-t border-slate-800">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto flex gap-3">
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
