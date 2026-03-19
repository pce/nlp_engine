import React from "react";
import Icon from "./Icon";

interface HeaderProps {
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
  active: (path: string) => string;
}

/**
 * Header Component
 * Handles the top-level navigation, sidebar toggling, and system status display.
 */
const Header: React.FC<HeaderProps> = ({ sidebarOpen, setSidebarOpen }) => {
  // Get current path name for the title, defaulting to "Dashboard"
  // const pageTitle = typeof window !== "undefined" ? window.location.pathname.substring(1) || "Dashboard" : "Dashboard";

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
          <div className="hidden md:flex flex-col items-end text-[10px] font-bold uppercase tracking-widest text-slate-400 border-r border-slate-200 dark:border-slate-700 pr-6">
            <span className="opacity-60">Engine Protocol</span>
            <span className="text-emerald-500 flex items-center gap-1">
              <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse"></span>
              API
            </span>
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
