import { useState, useEffect, useCallback, useRef } from "react";
import { nlpService, NLPRequest, StreamChunk } from "../services/nlp-service";
import { DocumentModel, DocumentState } from "../models/document";
import Icon from "./Icon";
import AnalysisDashboard from "./analysis/AnalysisDashboard";

interface DocumentPanelProps {
  content: string;
  outputContent?: string;
  onContentChange?: (content: string) => void;
  onOutputChange?: (content: string) => void;
  onAnalysisResultsRef?: React.MutableRefObject<((results: string) => void) | null>;
  isGenerating?: boolean;
}

const DocumentPanel = ({
  content,
  outputContent: externalOutputContent,
  onContentChange,
  onOutputChange,
  onAnalysisResultsRef,
  isGenerating,
}: DocumentPanelProps) => {
  // Initialize document state using the model helper
  const [doc, setDoc] = useState<DocumentState>(() => DocumentModel.createInitialState("Analysis Workspace", content));
  const [selectedText, setSelectedText] = useState<string>("");

  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [results, setResults] = useState<string>("");
  const [internalOutputContent, setInternalOutputContent] = useState<string>("");
  const [activeTab, setActiveTab] = useState<"editor" | "output" | "results">("editor");

  const outputContent = externalOutputContent !== undefined ? externalOutputContent : internalOutputContent;
  const setOutputContent = onOutputChange || setInternalOutputContent;
  const streamCleanupRef = useRef<(() => void) | null>(null);

  // Derived stats using the model logic
  const stats = DocumentModel.getStats(doc);

  const handleContentChange = (newContent: string) => {
    const updatedDoc = DocumentModel.updateContent(doc, newContent);
    setDoc(updatedDoc);
    onContentChange?.(newContent);
  };

  const handleSelection = (e: React.SyntheticEvent<HTMLTextAreaElement>) => {
    const target = e.target as HTMLTextAreaElement;
    const selection = target.value.substring(target.selectionStart, target.selectionEnd);
    setSelectedText(selection);
  };

  const handleProcessText = async () => {
    const textToProcess = selectedText || doc.content;
    if (!textToProcess.trim()) return;

    // Switch to results tab to show streaming analysis
    setActiveTab("results");
    setIsProcessing(true);
    setResults("");

    try {
      const request: NLPRequest = {
        text: textToProcess,
        plugin: "default",
        streaming: true,
        options: {
          pos_tagging: "true",
          terminology: "true",
        },
      };

      const cleanup = await nlpService.streamNLP(
        request,
        (chunk: StreamChunk) => {
          setResults((prev) => prev + chunk.chunk);
          if (chunk.is_final) {
            setIsProcessing(false);
          }
        },
        (error) => {
          console.error("Linguistic streaming error:", error);
          setIsProcessing(false);
          setResults((prev) => prev + "\n\n[Error: Processing failed]");
        },
      );

      streamCleanupRef.current = cleanup;
    } catch (error) {
      console.error("Submission error:", error);
      setIsProcessing(false);
    }
  };

  const handleClear = () => {
    if (activeTab === "editor") {
      handleContentChange("");
    } else if (activeTab === "output") {
      setOutputContent("");
    } else {
      setResults("");
    }

    if (streamCleanupRef.current) {
      streamCleanupRef.current();
      streamCleanupRef.current = null;
    }
  };

  const handleSave = () => {
    const dataToSave = activeTab === "output" ? outputContent : doc.content;
    localStorage.setItem(
      "nlp-studio-doc",
      JSON.stringify({
        ...DocumentModel.toJSON(doc),
        content: dataToSave,
      }),
    );
  };

  const handleLoad = () => {
    const saved = localStorage.getItem("nlp-studio-doc");
    if (saved) {
      try {
        const data = JSON.parse(saved);
        const loadedDoc = DocumentModel.fromJSON(data);
        setDoc(loadedDoc);
        onContentChange?.(loadedDoc.content);
      } catch (e) {
        console.error("Failed to hydrate document", e);
      }
    }
  };

  // Expose setResults to parent via ref
  useEffect(() => {
    if (onAnalysisResultsRef) {
      onAnalysisResultsRef.current = (newResults: string) => {
        setActiveTab("results");
        setResults(newResults);
        setIsProcessing(false);
      };
    }
    return () => {
      if (onAnalysisResultsRef) onAnalysisResultsRef.current = null;
    };
  }, [onAnalysisResultsRef]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamCleanupRef.current) {
        streamCleanupRef.current();
      }
    };
  }, []);

  // Listen for Markov generation updates from the Header
  useEffect(() => {
    if (isGenerating) {
      // Switch to Output tab when generation starts
      setActiveTab("output");
    }
  }, [isGenerating]);

  // Sync Input Source if content changed while on editor tab (initial load or manual sync)
  useEffect(() => {
    if (activeTab === "editor" && content !== doc.content && !isGenerating && !content.startsWith("Initializing")) {
      setDoc(DocumentModel.updateContent(doc, content));
    }
  }, [content, activeTab, isGenerating]);

  return (
    <div
      className="bg-white dark:bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-200 dark:border-slate-700 shadow-xl mb-6 overflow-hidden"
      style={{ backgroundColor: "var(--theme-surface)", borderColor: "var(--theme-border)" }}
    >
      <div
        className="p-4 bg-slate-50/50 dark:bg-slate-900/50 border-b border-slate-200 dark:border-slate-700"
        style={{ backgroundColor: "var(--theme-bg)", borderBottomColor: "var(--theme-border)", opacity: 0.8 }}
      >
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-indigo-100 dark:bg-indigo-900/30 rounded-lg" style={{ backgroundColor: "var(--theme-bg)" }}>
              <Icon name="document" size="sm" class="text-indigo-600 dark:text-indigo-400" style={{ color: "var(--theme-primary)" }} />
            </div>
            <div>
              <h2 className="text-sm font-black uppercase tracking-widest text-slate-800 dark:text-slate-200" style={{ color: "var(--theme-text)" }}>
                {doc.title}
              </h2>
              <div className="text-[9px] font-bold text-slate-400 uppercase tracking-tight" style={{ color: "var(--theme-text-muted)" }}>
                {activeTab === "editor" ? "Source Editor" : activeTab === "output" ? "Native Markov Output" : "Linguistic Analysis"}
              </div>
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleLoad}
              className="p-2 hover:bg-white dark:hover:bg-slate-700 rounded-lg transition-all border border-transparent hover:border-slate-200 dark:hover:border-slate-600"
              title="Load saved work"
            >
              <Icon name="import" size="sm" class="text-slate-500" />
            </button>
            <button
              onClick={handleSave}
              className="p-2 hover:bg-white dark:hover:bg-slate-700 rounded-lg transition-all border border-transparent hover:border-slate-200 dark:hover:border-slate-600"
              title="Save locally"
            >
              <Icon name="copy" size="sm" class="text-slate-500" />
            </button>
          </div>
        </div>
      </div>

      <div className="p-1">
        <div className="flex gap-1 bg-slate-100/50 dark:bg-slate-900/30 p-1 rounded-xl m-2" style={{ backgroundColor: "var(--theme-bg)" }}>
          <button
            className={`flex-1 py-2 text-[10px] font-black uppercase tracking-widest rounded-lg transition-all ${
              activeTab === "editor" ? "bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-300 shadow-sm" : "text-slate-500 hover:text-slate-700"
            }`}
            style={activeTab === "editor" ? { backgroundColor: "var(--theme-surface)", color: "var(--theme-primary)" } : { color: "var(--theme-text-muted)" }}
            onClick={() => setActiveTab("editor")}
          >
            Input Source
          </button>
          <button
            className={`flex-1 py-2 text-[10px] font-black uppercase tracking-widest rounded-lg transition-all ${
              activeTab === "output" ? "bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-300 shadow-sm" : "text-slate-500 hover:text-slate-700"
            }`}
            style={activeTab === "output" ? { backgroundColor: "var(--theme-surface)", color: "var(--theme-primary)" } : { color: "var(--theme-text-muted)" }}
            onClick={() => setActiveTab("output")}
          >
            Markov Output
          </button>
          <button
            className={`flex-1 py-2 text-[10px] font-black uppercase tracking-widest rounded-lg transition-all ${
              activeTab === "results" ? "bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-300 shadow-sm" : "text-slate-500 hover:text-slate-700"
            }`}
            style={activeTab === "results" ? { backgroundColor: "var(--theme-surface)", color: "var(--theme-primary)" } : { color: "var(--theme-text-muted)" }}
            onClick={() => setActiveTab("results")}
          >
            Analysis View
          </button>
        </div>

        <div className="p-4">
          {activeTab === "editor" && (
            <div className="animate-in fade-in duration-300 relative">
              <textarea
                value={doc.content}
                onChange={(e) => handleContentChange(e.target.value)}
                onSelect={handleSelection}
                className="w-full h-80 p-6 bg-transparent border-none text-lg leading-relaxed focus:outline-none focus:ring-0 resize-none font-serif placeholder:text-slate-300 dark:placeholder:text-slate-700 overflow-y-auto scrollbar-thin"
                style={{ color: "var(--theme-text)" }}
                placeholder="Start typing your document for C++ linguistic processing..."
                spellCheck="false"
              />
              <div
                className="mt-4 pt-4 border-t border-slate-100 dark:border-slate-800 flex justify-between items-center"
                style={{ borderTopColor: "var(--theme-border)" }}
              >
                <div
                  className="flex items-center gap-4 text-[10px] font-bold text-slate-400 uppercase tracking-tighter"
                  style={{ color: "var(--theme-text-muted)" }}
                >
                  <span>{stats.wordCount} Words</span>
                  <span>{stats.charCount} Chars</span>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={handleClear}
                    className="px-4 py-2 text-[10px] font-black uppercase tracking-widest text-slate-400 hover:text-red-500 transition-colors"
                  >
                    Clear
                  </button>
                  <button
                    onClick={handleProcessText}
                    disabled={isProcessing || (!doc.content.trim() && !selectedText.trim())}
                    className={`px-6 py-2 rounded-xl text-[10px] font-black uppercase tracking-[0.2em] transition-all shadow-lg active:scale-95 ${
                      isProcessing || (!doc.content.trim() && !selectedText.trim())
                        ? "bg-slate-100 dark:bg-slate-800 text-slate-400"
                        : "bg-indigo-600 text-white hover:bg-indigo-700"
                    }`}
                    style={!isProcessing && (doc.content.trim() || selectedText.trim()) ? { backgroundColor: "var(--theme-primary)" } : {}}
                  >
                    Analyze Source
                  </button>
                </div>
              </div>
            </div>
          )}

          {activeTab === "output" && (
            <div className="animate-in fade-in duration-300 relative">
              {isGenerating && (
                <div className="absolute top-4 right-4 z-10 flex items-center gap-2 bg-indigo-500 text-white px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest animate-pulse shadow-lg">
                  <div className="w-1.5 h-1.5 bg-white rounded-full animate-ping" />
                  Streaming from C++
                </div>
              )}
              <textarea
                readOnly
                value={outputContent}
                className="w-full h-80 p-6 bg-slate-50/30 dark:bg-slate-900/30 border-none text-lg leading-relaxed focus:outline-none focus:ring-0 resize-none font-serif placeholder:text-slate-300 dark:placeholder:text-slate-700 overflow-y-auto scrollbar-thin rounded-xl shadow-inner"
                style={{ color: "var(--theme-text)", backgroundColor: "var(--theme-bg)", opacity: 0.9 }}
                placeholder="Markov generation output will appear here..."
              />
              <div
                className="mt-4 pt-4 border-t border-slate-100 dark:border-slate-800 flex justify-between items-center"
                style={{ borderTopColor: "var(--theme-border)" }}
              >
                <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest" style={{ color: "var(--theme-text-muted)" }}>
                  {outputContent.split(/\s+/).filter(Boolean).length} Generated Words
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => setOutputContent("")}
                    className="px-4 py-2 text-[10px] font-black uppercase tracking-widest text-slate-400 hover:text-red-500 transition-colors"
                  >
                    Reset
                  </button>
                  <button
                    onClick={() => handleContentChange(outputContent)}
                    className="px-6 py-2 rounded-xl text-[10px] font-black uppercase tracking-[0.2em] bg-emerald-600 text-white hover:bg-emerald-700 transition-all shadow-lg active:scale-95"
                  >
                    Use as Input
                  </button>
                </div>
              </div>
            </div>
          )}

          {activeTab === "results" && (
            <div className="animate-in slide-in-from-bottom-2 duration-300 space-y-4">
              <div className="flex justify-between items-center px-2">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-indigo-500 rounded-full animate-pulse" style={{ backgroundColor: "var(--theme-primary)" }} />
                  <span className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400" style={{ color: "var(--theme-text-muted)" }}>
                    Engine Insight
                  </span>
                </div>
                <button
                  onClick={() => setResults("")}
                  className="text-[10px] font-black uppercase tracking-widest text-slate-400 hover:text-indigo-600 transition-colors"
                  style={{ color: "var(--theme-text-muted)" }}
                >
                  Clear Results
                </button>
              </div>
              <div
                className="bg-slate-50/50 dark:bg-slate-900/50 rounded-3xl p-6 min-h-[400px] overflow-hidden border border-slate-100 dark:border-slate-800 shadow-inner"
                style={{ backgroundColor: "var(--theme-bg)", borderColor: "var(--theme-border)" }}
              >
                <AnalysisDashboard results={results} isProcessing={isProcessing} />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DocumentPanel;
