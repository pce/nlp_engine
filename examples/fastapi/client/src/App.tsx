import React from "react";
import Dashboard from "./components/Dashboard";

import "./index.css";

import logo from "./logo.svg";
import reactLogo from "./react.svg";

class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  override componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("Uncaught error:", error, errorInfo);
  }

  override render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-slate-900 text-white p-8 flex flex-col items-center justify-center font-sans">
          <div className="max-w-2xl w-full bg-slate-800 border border-rose-500/30 rounded-3xl p-8 shadow-2xl">
            <h1 className="text-2xl font-black text-rose-500 mb-4 uppercase tracking-tighter">
              Application Error
            </h1>
            <p className="text-slate-400 mb-6 text-sm">
              The React engine encountered a critical rendering error. This
              usually happens due to missing components or data structure
              mismatches.
            </p>
            <div className="bg-black/50 p-6 rounded-xl font-mono text-xs text-rose-300/80 overflow-auto max-h-64 mb-6 border border-white/5">
              {this.state.error?.toString()}
            </div>
            <button
              onClick={() => window.location.reload()}
              className="px-6 py-3 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl text-xs font-black uppercase tracking-widest transition-all"
            >
              Reload Interface
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export function App() {
  return (
    <ErrorBoundary>
      <div className="">
        <Dashboard />
      </div>
    </ErrorBoundary>
  );
}

export default App;
