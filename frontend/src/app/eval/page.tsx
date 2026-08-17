'use client';
import { useState } from 'react';
import ParticleCanvas from '@/components/ParticleCanvas';
import Link from 'next/link';

interface EvalResult {
  question: string;
  faithfulness: number;
  relevancy: number;
  context_precision: number;
  context_recall: number;
}

interface EvalResponse {
  results: EvalResult[];
  averages: Record<string, number>;
}

export default function EvalPage() {
  const [data, setData] = useState<EvalResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

  const runEval = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${apiUrl}/api/eval/run`, {
        method: 'POST'
      });
      if (!res.ok) throw new Error('Failed to run evaluation');
      const json = await res.json();
      setData(json);
    } catch (err: any) {
      setError(err.message || 'An error occurred while evaluating');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen text-slate-50 font-sans flex flex-col">
      <ParticleCanvas />
      
      {/* Header */}
      <header className="relative z-10 p-6 border-b border-slate-800 bg-slate-900/50 backdrop-blur-md flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-pink-400">Golden Set Evaluation</h1>
        </div>
        <Link href="/" className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition-colors text-sm font-medium border border-slate-700">
          ← Back Home
        </Link>
      </header>
      
      <main className="relative z-10 flex-1 p-6 max-w-6xl mx-auto w-full flex flex-col gap-8 mt-4">
        
        <div className="flex flex-col md:flex-row justify-between items-center bg-slate-800/50 border border-slate-700 p-6 rounded-2xl gap-4">
          <div>
            <h2 className="text-lg font-semibold mb-1">Run Golden Set Evaluation</h2>
            <p className="text-slate-400 text-sm">This runs the entire test set through the RAG pipeline and computes faithfulness, relevancy, and context metrics.</p>
          </div>
          <button 
            onClick={runEval}
            disabled={loading}
            className="px-6 py-3 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium whitespace-nowrap"
          >
            {loading ? 'Evaluating...' : 'Start Evaluation'}
          </button>
        </div>

        {error && (
          <div className="bg-red-500/20 text-red-200 border border-red-500/30 p-4 rounded-lg">
            {error}
          </div>
        )}

        {data && (
          <div className="flex flex-col gap-8">
            <div className="bg-slate-800/50 border border-slate-700 p-6 rounded-2xl">
              <h3 className="text-xl font-semibold mb-6 text-indigo-300">Averages</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(data.averages).map(([k, v]) => (
                  <div key={k} className="bg-slate-900/50 p-4 rounded-xl border border-slate-700/50">
                    <div className="text-xs text-slate-400 mb-1 capitalize">{k.replace('_', ' ')}</div>
                    <div className="text-2xl font-bold text-slate-200">{v.toFixed(2)}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-slate-800/50 border border-slate-700 p-6 rounded-2xl overflow-x-auto">
              <h3 className="text-xl font-semibold mb-6 text-indigo-300">Per-Question Results</h3>
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-slate-700 text-slate-400">
                    <th className="pb-3 px-2">Question</th>
                    <th className="pb-3 px-2">Faithfulness</th>
                    <th className="pb-3 px-2">Relevancy</th>
                    <th className="pb-3 px-2">Ctx Precision</th>
                    <th className="pb-3 px-2">Ctx Recall</th>
                  </tr>
                </thead>
                <tbody>
                  {data.results.map((r, i) => (
                    <tr key={i} className="border-b border-slate-700/50 hover:bg-slate-700/20">
                      <td className="py-3 px-2">{r.question}</td>
                      <td className="py-3 px-2">{r.faithfulness.toFixed(2)}</td>
                      <td className="py-3 px-2">{r.relevancy.toFixed(2)}</td>
                      <td className="py-3 px-2">{r.context_precision.toFixed(2)}</td>
                      <td className="py-3 px-2">{r.context_recall.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
