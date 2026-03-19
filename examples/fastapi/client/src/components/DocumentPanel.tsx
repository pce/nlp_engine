import { useState, useEffect, useCallback, useRef } from "react";
import { nlpService, NLPRequest, StreamChunk } from "../services/nlp-service";
import { DocumentModel, DocumentState } from "../models/document";
import Icon from "./Icon";

interface DocumentPanelProps {
  onContentChange?: (content: string) => void;
}

const DocumentPanel = ({ onContentChange }: DocumentPanelProps) => {
  // Initialize document state using the model helper
  const [doc, setDoc] = useState<DocumentState>(() => DocumentModel.createInitialState("Analysis Workspace", ""));
  const [selectedText, setSelectedText] = useState<string>("");

  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [results, setResults] = useState<string>("");
  const [activeTab, setActiveTab] = useState<"editor" | "results">("editor");
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

    // Switch to results tab to show streaming
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

      // The new service API first submits a task then connects to SSE
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
    handleContentChange("");
    setResults("");
    if (streamCleanupRef.current) {
      streamCleanupRef.current();
      streamCleanupRef.current = null;
    }
  };

  const handleSave = () => {
    localStorage.setItem("nlp-studio-doc", JSON.stringify(DocumentModel.toJSON(doc)));
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

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamCleanupRef.current) {
        streamCleanupRef.current();
      }
    };
  }, []);

  return (
    <div className="bg-white dark:bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-200 dark:border-slate-700 shadow-xl mb-6 overflow-hidden">
      <div className="p-4 bg-slate-50/50 dark:bg-slate-900/50 border-b border-slate-200 dark:border-slate-700">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-indigo-100 dark:bg-indigo-900/30 rounded-lg">
              <Icon name="document" size="sm" class="text-indigo-600 dark:text-indigo-400" />
            </div>
            <h2 className="text-sm font-black uppercase tracking-widest text-slate-800 dark:text-slate-200">{doc.title}</h2>
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
        <div className="flex gap-1 bg-slate-100/50 dark:bg-slate-900/30 p-1 rounded-xl m-2">
          <button
            className={`flex-1 py-2 text-[10px] font-black uppercase tracking-widest rounded-lg transition-all ${
              activeTab === "editor" ? "bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-300 shadow-sm" : "text-slate-500 hover:text-slate-700"
            }`}
            onClick={() => setActiveTab("editor")}
          >
            Input Source
          </button>
          <button
            className={`flex-1 py-2 text-[10px] font-black uppercase tracking-widest rounded-lg transition-all ${
              activeTab === "results" ? "bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-300 shadow-sm" : "text-slate-500 hover:text-slate-700"
            }`}
            onClick={() => setActiveTab("results")}
          >
            Analysis View
          </button>
        </div>

        <div className="p-4">
          {activeTab === "editor" ? (
            <div className="animate-in fade-in duration-300">
              <textarea
                value={doc.content}
                onChange={(e) => handleContentChange(e.target.value)}
                onSelect={handleSelection}
                className="w-full h-80 p-6 bg-transparent border-none text-lg leading-relaxed focus:outline-none focus:ring-0 resize-none font-serif placeholder:text-slate-300 dark:placeholder:text-slate-700 overflow-y-auto scrollbar-thin"
                placeholder="Start typing your document for C++ linguistic processing..."
                spellCheck="false"
              />
              <div className="mt-4 pt-4 border-t border-slate-100 dark:border-slate-800 flex justify-between items-center">
                <div className="flex items-center gap-4 text-[10px] font-bold text-slate-400 uppercase tracking-tighter">
                  <span>{stats.wordCount} Words</span>
                  <span>{stats.charCount} Chars</span>
                  {selectedText && (
                    <span className="bg-indigo-50 dark:bg-indigo-900/20 text-indigo-600 dark:text-indigo-400 px-2 py-0.5 rounded">
                      {selectedText.length} Chars Selected
                    </span>
                  )}
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
                        : "bg-indigo-600 text-white hover:bg-indigo-700 shadow-indigo-500/20"
                    }`}
                  >
                    {isProcessing ? "Linguistic Bridge Active..." : selectedText.trim() ? "Analyze Selection" : "Run C++ Engine"}
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="animate-in slide-in-from-bottom-2 duration-300">
              <div className="bg-slate-50/50 dark:bg-slate-900/50 rounded-2xl p-6 min-h-[320px] max-h-[500px] overflow-y-auto border border-slate-100 dark:border-slate-800">
                {results ? (
                  <div className="whitespace-pre-wrap text-sm font-mono leading-relaxed text-slate-700 dark:text-slate-300">
                    {results}
                    {isProcessing && <span className="inline-block w-2 h-4 ml-1 bg-indigo-500 animate-pulse" />}
                  </div>
                ) : (
                  <div className="h-64 flex flex-col items-center justify-center text-slate-400 space-y-4">
                    <div className="w-12 h-12 bg-slate-100 dark:bg-slate-800 rounded-full flex items-center justify-center opacity-50">
                      <Icon name="analytics" size="md" />
                    </div>
                    <p className="text-[10px] font-black uppercase tracking-widest">Awaiting Engine Results</p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DocumentPanel;
