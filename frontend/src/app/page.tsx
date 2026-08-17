import ParticleCanvas from "@/components/ParticleCanvas";
import Link from "next/link";

export default function Home() {
  return (
    <main className="relative min-h-screen text-slate-50 overflow-hidden font-sans">
      <ParticleCanvas />
      
      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-4 py-20 text-center">
        
        {/* Title */}
        <div className="mb-4 inline-flex items-center justify-center">
          <h1 className="text-6xl md:text-8xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 pb-2">
            PinnacleRAG-DS
          </h1>
        </div>

        <h2 className="text-4xl md:text-6xl font-bold tracking-tight mb-6 bg-clip-text text-transparent bg-gradient-to-br from-white via-slate-200 to-slate-500">
          The Ultimate Hybrid RAG
        </h2>

        {/* Domain Selection Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-5xl">
          {/* Trading Card */}
          <Link href="/chat?domain=trading" className="group relative rounded-2xl border border-slate-700 bg-slate-800/40 backdrop-blur-sm p-8 hover:bg-slate-800/60 transition-all duration-300 hover:-translate-y-1 hover:shadow-2xl hover:shadow-blue-500/20 cursor-pointer overflow-hidden text-left block">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <h3 className="text-2xl font-semibold mb-3 text-blue-400">Trading & Finance</h3>
            <p className="text-slate-400 text-sm mb-6">Strict financial grounding, 10-K analysis, and mandated risk disclaimers.</p>
            <div className="text-blue-500 font-medium group-hover:translate-x-2 transition-transform inline-flex items-center">
              Launch Agent →
            </div>
          </Link>

          {/* Security Card */}
          <Link href="/chat?domain=security" className="group relative rounded-2xl border border-slate-700 bg-slate-800/40 backdrop-blur-sm p-8 hover:bg-slate-800/60 transition-all duration-300 hover:-translate-y-1 hover:shadow-2xl hover:shadow-red-500/20 cursor-pointer overflow-hidden text-left block">
            <div className="absolute inset-0 bg-gradient-to-br from-red-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <h3 className="text-2xl font-semibold mb-3 text-red-400">Cyber Security</h3>
            <p className="text-slate-400 text-sm mb-6">Threat analysis, CVE mapping, and strict anti-hallucination guardrails.</p>
            <div className="text-red-500 font-medium group-hover:translate-x-2 transition-transform inline-flex items-center">
              Launch Agent →
            </div>
          </Link>

          {/* SEO Card */}
          <Link href="/chat?domain=seo" className="group relative rounded-2xl border border-slate-700 bg-slate-800/40 backdrop-blur-sm p-8 hover:bg-slate-800/60 transition-all duration-300 hover:-translate-y-1 hover:shadow-2xl hover:shadow-emerald-500/20 cursor-pointer overflow-hidden text-left block">
            <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <h3 className="text-2xl font-semibold mb-3 text-emerald-400">Technical SEO</h3>
            <p className="text-slate-400 text-sm mb-6">Keyword strategies, gap analysis, and content recommendation engine.</p>
            <div className="text-emerald-500 font-medium group-hover:translate-x-2 transition-transform inline-flex items-center">
              Launch Agent →
            </div>
          </Link>
        </div>
        
        {/* Eval Link */}
        <div className="mt-12">
          <Link href="/eval" className="inline-flex items-center gap-2 px-6 py-3 rounded-full bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 hover:text-white transition-all">
            <span className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse"></span>
            Golden Set Evaluation Dashboard
          </Link>
        </div>

      </div>
    </main>
  );
}
