import React from "react";
import { Moon, Sparkles, Activity, ShieldCheck } from "lucide-react";

interface TopNavbarProps {
  systemStatus: "idle" | "processing" | "completed" | "error";
  isMuted: boolean;
  setIsMuted: (muted: boolean) => void;
}

export const TopNavbar: React.FC<TopNavbarProps> = ({ systemStatus, isMuted, setIsMuted }) => {
  const getStatusBadge = () => {
    switch (systemStatus) {
      case "processing":
        return (
          <div className="flex items-center space-x-2 bg-accentPurple/10 border border-accentPurple/20 px-3 py-1 rounded-full text-xs text-accentPurple-light">
            <Activity className="w-3.5 h-3.5 animate-spin" />
            <span className="font-mono">INFERENCE_RUNNING</span>
          </div>
        );
      case "completed":
        return (
          <div className="flex items-center space-x-2 bg-emerald-500/10 border border-emerald-500/20 px-3 py-1 rounded-full text-xs text-emerald-400">
            <ShieldCheck className="w-3.5 h-3.5" />
            <span className="font-mono">VALIDATED_PASS</span>
          </div>
        );
      case "error":
        return (
          <div className="flex items-center space-x-2 bg-red-500/10 border border-red-500/20 px-3 py-1 rounded-full text-xs text-red-400">
            <div className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse" />
            <span className="font-mono">FAIL_CLOSED_HALT</span>
          </div>
        );
      default:
        return (
          <div className="flex items-center space-x-2 bg-white/5 border border-white/10 px-3 py-1 rounded-full text-xs text-gray-400">
            <div className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
            <span className="font-mono">READY_STANDBY</span>
          </div>
        );
    }
  };

  return (
    <header className="fixed top-4 left-4 right-4 h-14 z-50 glass-card rounded-2xl border border-white/5 px-6 flex items-center justify-between">
      {/* Left side logo and title */}
      <div className="flex items-center space-x-3">
        <Sparkles className="w-5 h-5 text-accentPurple animate-pulse" />
        <div className="flex items-center space-x-2">
          <h1 className="text-lg font-bold font-heading tracking-tight text-white m-0">
            VIDEOCAP
          </h1>
          <span className="bg-gradient-to-r from-accentPurple to-accentCyan text-white text-[9px] px-2 py-0.5 rounded font-mono font-bold uppercase tracking-wider">
            TRACK 2
          </span>
        </div>
      </div>

      {/* Middle section status badge */}
      <div className="hidden md:flex items-center space-x-4">
        {getStatusBadge()}
      </div>

      {/* Right side controls */}
      <div className="flex items-center space-x-4">
        {/* Sync / Sound toggle */}
        <button
          onClick={() => setIsMuted(!isMuted)}
          className="text-xs font-mono text-gray-400 hover:text-white transition-all bg-white/5 hover:bg-white/10 px-3 py-1.5 rounded-lg border border-white/5"
        >
          {isMuted ? "🔇 AUDIO_MUTED" : "🔊 AUDIO_SYNCED"}
        </button>

        {/* Fake Theme toggle */}
        <button className="p-2 text-gray-400 hover:text-white hover:bg-white/5 rounded-xl transition-all" aria-label="Toggle Theme">
          <Moon className="w-4 h-4" />
        </button>

        {/* GitHub link */}
        <a
          href="https://github.com"
          target="_blank"
          rel="noreferrer"
          className="p-2 text-gray-400 hover:text-white hover:bg-white/5 rounded-xl transition-all"
          aria-label="GitHub Repository"
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
            <path d="M9 18c-4.51 2-5-2-7-2" />
          </svg>
        </a>
      </div>
    </header>
  );
};
