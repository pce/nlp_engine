import React, { useState, useRef, useEffect } from "react";
import Icon from "./Icon";
import { nlpService } from "../services/nlp-service";

interface HeaderProps {
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
  active: (path: string) => string;
  onContentChange?: (content: string) => void;
}

/**
 * Header Component
 * Handles the top-level navigation, sidebar toggling, and system status display.
 */
const Header: React.FC<HeaderProps> = ({ sidebarOpen, setSidebarOpen, onContentChange }) => {
  const [isMarkovOpen, setIsMarkovOpen] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("generic_novel");
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Fetch available models on mount
  useEffect(() => {
    nlpService.getAvailableModels().then((models) => {
      if (models && models.length > 0) {
        setAvailableModels(models);
        // Default to generic_novel if it exists in the list
        if (models.includes("generic_novel")) {
          setSelectedModel("generic_novel");
        } else {
          setSelectedModel(models[0]);
        }
      }
    });
  }, []);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsMarkovOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleGenerate = async (withInput: boolean) => {
    setIsMarkovOpen(false);
    setIsGenerating(true);

    try {
      const textarea = document.querySelector("textarea");
      let seed = "";
      let currentFullText = "";

      if (textarea) {
        currentFullText = textarea.value;
        if (withInput) {
          const start = textarea.selectionStart;
          const end = textarea.selectionEnd;
          seed = start !== end ? currentFullText.substring(start, end) : currentFullText.slice(-100);
        }
      }

      const response = await nlpService.generateMarkov({
        seed: seed || "The",
        length: 150,
        model: selectedModel,
      });

      if (response && response.output && onContentChange) {
        // The previous bug was likely due to grabbing DOM state that included
        // stale analysis stream results if the user switched tabs rapidly.
        // We ensure we only ever work with the actual editor content.
        if (withInput && currentFullText) {
          const separator = currentFullText.length > 0 && !currentFullText.endsWith(" ") ? " " : "";
          onContentChange(currentFullText + separator + response.output);
        } else {
          onContentChange(response.output);
        }
      }
    } catch (error) {
      console.error("Failed to generate Markov text:", error);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <header className="bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 p-4 sticky top-0 z-40">
      <div className="flex items-center justify-between">
        <div className="flex items-center">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
            aria-label="Toggle Sidebar"
          >
            <Icon name="columns" size="sm" class="text-slate-600 dark:text-slate-400" />
          </button>

          {/*<div className="ml-2 md:ml-0 flex items-center gap-2">
            <div className="w-8 h-8 bg-indigo-600 rounded flex items-center justify-center text-white font-black shadow-lg">N</div>
            <h2 className="text-xl font-black tracking-tight text-slate-800 dark:text-white capitalize leading-none">{pageTitle}</h2>
          </div>*/}
        </div>

        <div className="flex items-center gap-6">
          <div className="relative" ref={dropdownRef}>
            <button
              onClick={() => setIsMarkovOpen(!isMarkovOpen)}
              className={`group flex items-center gap-3 px-4 py-2 rounded-xl border border-slate-200 dark:border-slate-700 transition-all ${
                isMarkovOpen
                  ? "bg-indigo-50 dark:bg-indigo-900/20 border-indigo-200 dark:border-indigo-800 shadow-sm"
                  : "bg-white dark:bg-slate-800 hover:border-slate-300 dark:hover:border-slate-600"
              }`}
            >
              <div className="flex flex-col items-end text-[10px] font-black uppercase tracking-widest">
                <span className={`${isGenerating ? "text-amber-500" : "text-slate-400"} transition-colors`}>Markov Engine</span>
                <span className={`${isGenerating ? "text-amber-600 animate-pulse" : "text-indigo-600 dark:text-indigo-400"}`}>
                  {isGenerating ? "Processing..." : selectedModel.replace(/_/g, " ")}
                </span>
              </div>
              <div
                className={`p-1.5 rounded-lg transition-colors ${isMarkovOpen ? "bg-indigo-600 text-white" : "bg-slate-100 dark:bg-slate-700 text-slate-500"}`}
              >
                <Icon name={isGenerating ? "refresh" : "sparkles"} size="sm" className={isGenerating ? "animate-spin" : ""} />
              </div>
            </button>

            {isMarkovOpen && (
              <div className="absolute right-0 mt-3 w-64 bg-white dark:bg-slate-800 rounded-2xl shadow-2xl border border-slate-200 dark:border-slate-700 py-3 z-50 animate-in fade-in slide-in-from-top-4 duration-300">
                <div className="px-4 pb-2 mb-2 border-b border-slate-100 dark:border-slate-700">
                  <span className="text-[9px] font-black uppercase tracking-[0.2em] text-slate-400">Knowledge Pack</span>
                  <div className="mt-2 grid grid-cols-1 gap-1">
                    {availableModels.map((model) => (
                      <button
                        key={model}
                        onClick={() => setSelectedModel(model)}
                        className={`text-left px-3 py-2 rounded-lg text-[10px] font-bold transition-all ${
                          selectedModel === model
                            ? "bg-indigo-600 text-white shadow-md shadow-indigo-500/20"
                            : "text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-700"
                        }`}
                      >
                        {model.replace(/_/g, " ")}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="px-2 space-y-1">
                  <button
                    onClick={() => handleGenerate(false)}
                    disabled={isGenerating}
                    className="w-full text-left px-3 py-2.5 rounded-xl text-[10px] font-black uppercase tracking-widest text-slate-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-indigo-900/20 hover:text-indigo-600 transition-all flex items-center justify-between group"
                  >
                    <div className="flex items-center gap-2">
                      <Icon name="sparkles" size="sm" />
                      <span>Generate Story</span>
                    </div>
                    <Icon name="chevron-down" size="sm" className="-rotate-90 opacity-0 group-hover:opacity-100 transition-all" />
                  </button>
                  <button
                    onClick={() => handleGenerate(true)}
                    disabled={isGenerating}
                    className="w-full text-left px-3 py-2.5 rounded-xl text-[10px] font-black uppercase tracking-widest text-slate-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-indigo-900/20 hover:text-indigo-600 transition-all flex items-center justify-between group"
                  >
                    <div className="flex items-center gap-2">
                      <Icon name="edit" size="sm" />
                      <span>Continue from Text</span>
                    </div>
                    <Icon name="chevron-down" size="sm" className="-rotate-90 opacity-0 group-hover:opacity-100 transition-all" />
                  </button>
                </div>

                <div className="mt-3 px-4 pt-2 border-t border-slate-100 dark:border-slate-700">
                  <div className="flex items-center justify-between text-[8px] font-bold text-slate-400 uppercase">
                    <span>C++ Native Core</span>
                    <span className="flex items-center gap-1">
                      <span className="w-1 h-1 bg-emerald-500 rounded-full"></span>
                      Ready
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="flex items-center gap-3">
            <button className="p-2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors">
              <Icon name="settings" size="sm" />
            </button>
            <button className="w-9 h-9 rounded-xl bg-slate-100 dark:bg-slate-700 border border-slate-200 dark:border-slate-600 flex items-center justify-center hover:ring-2 ring-indigo-500/20 transition-all shadow-sm">
              <span className="text-xs font-black text-slate-600 dark:text-slate-300">JS</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
