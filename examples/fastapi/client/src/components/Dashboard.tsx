import React, { useState, useRef } from "react";
import Sidebar from "./Sidebar";
import Header from "./Header";
import DocumentPanel from "./DocumentPanel";
import Icon from "./Icon";

/**
 * Dashboard Component
 * The main layout container for the NLP Studio.
 * Manages shared state between the DocumentPanel (editor) and Sidebar (analysis).
 */
const Dashboard: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [documentContent, setDocumentContent] = useState("");
  const [outputContent, setOutputContent] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const analysisResultsRef = useRef<((results: string) => void) | null>(null);

  // active navigation styling (TODO expand with a router)
  const activeLink = (path: string) =>
    window.location.pathname === path
      ? "text-indigo-600 font-bold"
      : "text-slate-500 hover:text-indigo-500";

  return (
    <div
      className="flex h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 overflow-hidden"
      style={{ backgroundColor: "var(--theme-bg)", color: "var(--theme-text)" }}
    >
      {/* Sidebar - Positioned for desktop and mobile toggle */}
      <div
        className={`fixed inset-y-0 left-0 z-50 transform ${sidebarOpen ? "translate-x-0" : "-translate-x-full"} md:relative md:translate-x-0 ${
          sidebarOpen ? "md:w-[400px]" : "md:w-0"
        } transition-all duration-300 ease-in-out overflow-hidden`}
      >
        <div className="w-[400px] h-full">
          <Sidebar documentContent={documentContent} />
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden relative">
        <Header
          sidebarOpen={sidebarOpen}
          setSidebarOpen={setSidebarOpen}
          active={activeLink}
          onContentChange={(content, target) => {
            if (target === "output") {
              setOutputContent(content);
            } else {
              setDocumentContent(content);
            }
          }}
          onAnalysisResults={(results) => analysisResultsRef.current?.(results)}
          isGenerating={isGenerating}
          setIsGenerating={setIsGenerating}
        />

        <main className="flex-1 overflow-y-auto p-4 md:p-8 space-y-8 scrollbar-thin">
          <div className="max-w-5xl mx-auto space-y-8">
            {/* Hero Section */}
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
              <div className="space-y-1">
                <div
                  className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-100 dark:border-indigo-800 text-[10px] font-black uppercase tracking-widest text-indigo-600 dark:text-indigo-400"
                  style={{
                    backgroundColor: "var(--theme-bg)",
                    borderColor: "var(--theme-border)",
                    color: "var(--theme-primary)",
                  }}
                >
                  <span className="relative flex h-2 w-2">
                    <span
                      className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"
                      style={{ backgroundColor: "var(--theme-primary)" }}
                    ></span>
                    <span
                      className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"
                      style={{ backgroundColor: "var(--theme-primary)" }}
                    ></span>
                  </span>
                  C++ Native Interface
                </div>
                <h1
                  className="text-4xl font-black tracking-tight text-slate-800 dark:text-white leading-none mt-2"
                  style={{ color: "var(--theme-text)" }}
                >
                  NLP Studio <span className="font-light opacity-30">beta</span>
                </h1>
                <p
                  className="text-slate-500 dark:text-slate-400 text-sm font-medium"
                  style={{ color: "var(--theme-text-muted)" }}
                >
                  Perform high-performance linguistic analysis directly via
                  FastAPI bridge.
                </p>
              </div>
            </div>

            {/* Central Editor/Analysis Workspace */}
            <DocumentPanel
              content={documentContent}
              outputContent={outputContent}
              onContentChange={setDocumentContent}
              onOutputChange={setOutputContent}
              onAnalysisResultsRef={analysisResultsRef}
              isGenerating={isGenerating}
            />

            {/* TODO Quick Stats */}
          </div>
        </main>
      </div>

      {/* Mobile Sidebar Overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-slate-900/20 backdrop-blur-sm z-40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
    </div>
  );
};

export default Dashboard;
