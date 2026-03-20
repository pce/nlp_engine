import React from "react";
import Icon from "../Icon";
import Dropdown from "../ui/Dropdown";

interface ToolkitDropdownProps {
  onAction: (method: string, options?: any) => void;
}

/**
 * ToolkitDropdown Component
 * Handles experimental text processing features like Deduplication.
 * Styled using theme variables for brand consistency.
 */
const ToolkitDropdown: React.FC<ToolkitDropdownProps> = ({ onAction }) => {
  return (
    <Dropdown label="Toolkit" subLabel="Experimental" icon="tool" variant="warning">
      <div className="px-4 pb-2 mb-2 border-b" style={{ borderBottomColor: "var(--theme-border)" }}>
        <span className="text-[9px] font-black uppercase tracking-widest" style={{ color: "var(--theme-text-muted)" }}>
          Processing Tools
        </span>
      </div>
      <div className="px-2 space-y-1">
        <button
          onClick={() => onAction("deduplicator", { mode: "detect" })}
          className="w-full text-left px-3 py-2 rounded-lg text-[10px] font-bold flex items-center gap-2 transition-all hover:bg-slate-500/10"
          style={{ color: "var(--theme-text)" }}
        >
          <Icon name="search" size="sm" /> Find Duplicates
        </button>
        <button
          onClick={() => onAction("deduplicator", { mode: "remove" })}
          className="w-full text-left px-3 py-2 rounded-lg text-[10px] font-bold flex items-center gap-2 transition-all hover:bg-rose-500/10"
          style={{ color: "var(--theme-danger)" }}
        >
          <Icon name="trash" size="sm" /> Delete Duplicates
        </button>
      </div>
    </Dropdown>
  );
};

export default ToolkitDropdown;
