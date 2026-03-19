import React, { useMemo } from "react";
import Icon from "../Icon";

interface NLPStats {
  tokens?: number;
  sentences?: number;
  readability_score?: number;
  sentiment_score?: number;
  pos_distribution?: Record<string, number>;
  entities?: Array<{ text: string; label: string }>;
  [key: string]: any;
}

interface AnalysisDashboardProps {
  results: string;
  isProcessing: boolean;
}

/**
 * AnalysisDashboard Component
 * A scientific, "dashboardy" panel for visualizing NLP engine results.
 * Parses raw JSON/text output from the C++ engine into visual components.
 */
const AnalysisDashboard: React.FC<AnalysisDashboardProps> = ({ results, isProcessing }) => {
  // Parse results safely. The engine might return raw text or a JSON string.
  const data = useMemo(() => {
    if (!results) return null;
    try {
      return JSON.parse(results) as NLPStats;
    } catch (e) {
      // If not JSON, it might be streaming raw text or partial data
      return null;
    }
  }, [results]);

  if (!results && !isProcessing) {
    return (
      <div className="h-64 flex flex-col items-center justify-center text-slate-400 space-y-4 animate-in fade-in duration-500">
        <div className="w-16 h-16 bg-slate-100 dark:bg-slate-800 rounded-2xl flex items-center justify-center opacity-50">
          <Icon name="analytics" size="lg" />
        </div>
        <p className="text-[10px] font-black uppercase tracking-[0.3em]">Awaiting Engine Analysis</p>
      </div>
    );
  }

  // Fallback for raw text streaming (if it's not JSON yet)
  if (!data && results) {
    return (
      <div className="p-6 bg-slate-900 rounded-2xl border border-slate-800 font-mono text-xs text-emerald-400 leading-relaxed overflow-y-auto max-h-[500px] shadow-inner">
        <div className="flex items-center gap-2 mb-4 text-[10px] font-black uppercase tracking-widest text-emerald-500/50">
          <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
          Raw Stream Data
        </div>
        {results}
        {isProcessing && <span className="inline-block w-2 h-4 ml-1 bg-emerald-500 animate-pulse" />}
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-in slide-in-from-bottom-4 duration-500">
      {/* High Level Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          label="Sentiment"
          value={data?.sentiment_score?.toFixed(2) || "0.00"}
          icon="sentiment"
          color="text-blue-500"
          trend={data?.sentiment_score && data.sentiment_score > 0 ? "positive" : "negative"}
        />
        <StatCard
          label="Readability"
          value={data?.readability_score?.toFixed(1) || "0.0"}
          icon="readability"
          color="text-amber-500"
        />
        <StatCard
          label="Tokens"
          value={data?.tokens?.toString() || "0"}
          icon="brain"
          color="text-indigo-500"
        />
        <StatCard
          label="Sentences"
          value={data?.sentences?.toString() || "0"}
          icon="rows"
          color="text-emerald-500"
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* POS Distribution Histogram */}
        <div className="md:col-span-2 bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm">
          <h3 className="text-[10px] font-black uppercase tracking-widest text-slate-400 mb-6 flex items-center gap-2">
            <Icon name="stats" size="sm" />
            Linguistic Distribution
          </h3>
          <div className="space-y-4">
            {data?.pos_distribution ? (
              Object.entries(data.pos_distribution)
                .sort(([, a], [, b]) => (b as number) - (a as number))
                .slice(0, 6)
                .map(([tag, count], idx) => (
                  <div key={tag} className="space-y-1 group">
                    <div className="flex justify-between text-[10px] font-bold uppercase tracking-tighter">
                      <span className="text-slate-500 group-hover:text-indigo-600 transition-colors">{tag}</span>
                      <span className="text-slate-400">{count}</span>
                    </div>
                    <div className="h-1.5 w-full bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-indigo-500 rounded-full transition-all duration-1000 ease-out"
                        style={{ width: `${Math.min(100, ((count as number) / (data.tokens || 1)) * 200)}%`, transitionDelay: `${idx * 100}ms` }}
                      />
                    </div>
                  </div>
                ))
            ) : (
              <p className="text-xs text-slate-400 italic">No distribution data available</p>
            )}
          </div>
        </div>

        {/* Entities / Keywords Panel */}
        <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm">
          <h3 className="text-[10px] font-black uppercase tracking-widest text-slate-400 mb-6 flex items-center gap-2">
            <Icon name="sparkles" size="sm" />
            Key Terminology
          </h3>
          <div className="flex flex-wrap gap-2">
            {data?.entities && data.entities.length > 0 ? (
              data.entities.map((entity, i) => (
                <span
                  key={i}
                  className="px-2 py-1 rounded-md bg-indigo-50 dark:bg-indigo-900/20 text-indigo-600 dark:text-indigo-400 text-[10px] font-black uppercase border border-indigo-100 dark:border-indigo-800/50 hover:scale-105 transition-transform cursor-default"
                >
                  {entity.text}
                </span>
              ))
            ) : (
              <p className="text-xs text-slate-400 italic">Extracting entities...</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

interface StatCardProps {
  label: string;
  value: string;
  icon: any;
  color: string;
  trend?: "positive" | "negative";
}

const StatCard: React.FC<StatCardProps> = ({ label, value, icon, color, trend }) => (
  <div className="bg-white dark:bg-slate-900 p-4 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm hover:shadow-md transition-all group">
    <div className="flex items-center justify-between mb-2">
      <div className={`p-2 rounded-xl bg-slate-50 dark:bg-slate-800 group-hover:bg-indigo-50 dark:group-hover:bg-indigo-900/20 transition-colors ${color}`}>
        <Icon name={icon} size="sm" />
      </div>
      {trend && (
        <span className={`text-[8px] font-black px-1.5 py-0.5 rounded ${trend === "positive" ? "bg-emerald-100 text-emerald-700" : "bg-red-100 text-red-700"}`}>
          {trend === "positive" ? "↑" : "↓"}
        </span>
      )}
    </div>
    <div className="text-2xl font-black tracking-tight text-slate-800 dark:text-white leading-none">{value}</div>
    <div className="text-[9px] font-black uppercase tracking-widest text-slate-400 mt-1">{label}</div>
  </div>
);

export default AnalysisDashboard;
