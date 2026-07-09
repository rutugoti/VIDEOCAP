import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  UploadCloud,
  Layers,
  Eye,
  Volume2,
  Type,
  GitMerge,
  Network,
  BookOpen,
  Cpu,
  ShieldAlert,
  CheckCircle2,
} from "lucide-react";

export interface PipelineStage {
  id: string;
  name: string;
  desc: string;
  time: string;
  model: string;
  icon: any;
}

const STAGES: PipelineStage[] = [
  { id: "upload", name: "Upload", desc: "Ingests MP4/MKV video container & validates dimensions/duration", time: "0.2s", model: "Native FFmpeg", icon: UploadCloud },
  { id: "sampling", name: "Sampling", desc: "Uniform/complexity-aware adaptive frame sampling", time: "0.8s", model: "PyAV + Complexity", icon: Layers },
  { id: "vision", name: "Vision", desc: "VLM detects key frame contents, objects, and actions", time: "1.4s", model: "Llama-4-Scout-17b", icon: Eye },
  { id: "speech", name: "Speech", desc: "Transcribes audio tracks & maps timestamps", time: "2.1s", model: "Whisper-v3", icon: Volume2 },
  { id: "ocr", name: "OCR", desc: "Extracts video on-screen text overlays", time: "1.1s", model: "Llama-4-Scout-17b", icon: Type },
  { id: "fusion", name: "Fusion", desc: "Groups multi-modal observations into temporal events", time: "0.5s", model: "Llama-3.3-70b", icon: GitMerge },
  { id: "graph", name: "Graph", desc: "Constructs semantic relationship graphs of actors & actions", time: "0.2s", model: "NetworkX Builder", icon: Network },
  { id: "narrative", name: "Narrative", desc: "Generates factually-grounded, style-neutral text summary", time: "1.2s", model: "Llama-3.3-70b", icon: BookOpen },
  { id: "generation", name: "Captioner", desc: "Runs single-pass styled captions generation", time: "2.4s", model: "Llama-3.3-70b", icon: Cpu },
  { id: "validation", name: "Validation", desc: "Enforces pass/fail rules against hallucinations & drift", time: "1.6s", model: "Llama-3.3-70b", icon: ShieldAlert },
  { id: "submission", name: "Submission", desc: "Formats validated captions into final submission JSON", time: "0.1s", model: "Formatter", icon: CheckCircle2 },
];

interface PipelineProps {
  currentStage: string; // The current stage ID from the backend, e.g. "vision"
  status: string;       // The job status, e.g. "pending", "running", "completed", "failed"
}

export const Pipeline: React.FC<PipelineProps> = ({ currentStage, status }) => {
  const [hoveredStageId, setHoveredStageId] = useState<string | null>(null);

  const getStageStatus = (stageId: string, index: number) => {
    const currentIdx = STAGES.findIndex(s => s.id === currentStage);
    
    if (status === "failed" && stageId === currentStage) {
      return "error";
    }
    
    if (status === "completed") {
      return "completed";
    }
    
    if (status === "pending" || currentIdx === -1) {
      return "pending";
    }

    if (index < currentIdx) {
      return "completed";
    }
    
    if (stageId === currentStage) {
      return status === "completed" ? "completed" : "active";
    }
    
    return "pending";
  };

  return (
    <div className="w-full h-full glass-card rounded-2xl border border-white/5 p-4 flex flex-col justify-between overflow-hidden relative">
      {/* Title block */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          <div className="w-1.5 h-3.5 bg-accentPurple rounded-full" />
          <span className="text-xs font-bold uppercase tracking-wider text-gray-400">
            Engine Processing Pipeline
          </span>
        </div>
        <div className="text-[10px] font-mono text-gray-500">
          PROCESSED_VIA_LLAMA_NATIVE
        </div>
      </div>

      {/* Nodes and connectors container */}
      <div className="flex-1 flex items-center justify-between w-full overflow-x-auto custom-scrollbar px-2 space-x-1 lg:space-x-2 py-4">
        {STAGES.map((stage, idx) => {
          const nodeStatus = getStageStatus(stage.id, idx);
          const Icon = stage.icon;

          // Color themes based on status
          let borderClass = "border-white/10 text-gray-500 bg-white/2";
          let textClass = "text-gray-500";
          let glowClass = "";
          
          if (nodeStatus === "completed") {
            borderClass = "border-emerald-500/50 text-emerald-400 bg-emerald-500/10";
            textClass = "text-emerald-400";
            glowClass = "shadow-[0_0_15px_rgba(16,185,129,0.3)]";
          } else if (nodeStatus === "active") {
            borderClass = "border-accentPurple text-white bg-accentPurple/20";
            textClass = "text-white font-semibold";
            glowClass = "shadow-[0_0_20px_rgba(168,85,247,0.5)] animate-pulse";
          } else if (nodeStatus === "error") {
            borderClass = "border-red-500 text-red-400 bg-red-500/20";
            textClass = "text-red-400 font-semibold";
            glowClass = "shadow-[0_0_20px_rgba(239,68,68,0.5)] animate-ping";
          }

          return (
            <React.Fragment key={stage.id}>
              {/* Pipeline Node */}
              <div
                className="relative flex flex-col items-center select-none shrink-0"
                onMouseEnter={() => setHoveredStageId(stage.id)}
                onMouseLeave={() => setHoveredStageId(null)}
              >
                <motion.div
                  whileHover={{ scale: 1.1 }}
                  className={`w-12 h-12 rounded-xl flex items-center justify-center border cursor-pointer transition-all duration-300 ${borderClass} ${glowClass}`}
                >
                  <Icon className="w-5 h-5" />
                  {/* Status Indicator Dot */}
                  {nodeStatus === "active" && (
                    <span className="absolute -top-1 -right-1 w-3 h-3 bg-accentPurple rounded-full animate-ping" />
                  )}
                  {nodeStatus === "completed" && (
                    <span className="absolute -top-1 -right-1 w-3 h-3 bg-emerald-500 rounded-full flex items-center justify-center text-[8px] text-white font-bold">
                      ✓
                    </span>
                  )}
                </motion.div>
                
                <span className={`text-[10px] mt-2 font-mono tracking-tight ${textClass}`}>
                  {stage.name}
                </span>

                {/* Floating tooltip */}
                <AnimatePresence>
                  {hoveredStageId === stage.id && (
                    <motion.div
                      initial={{ opacity: 0, y: 10, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: 10, scale: 0.95 }}
                      transition={{ duration: 0.15 }}
                      className="absolute bottom-16 z-50 w-56 p-3 glass-card rounded-xl border border-white/10 text-xs shadow-2xl pointer-events-none"
                    >
                      <div className="font-bold text-white mb-1">{stage.name}</div>
                      <p className="text-gray-400 mb-2 leading-relaxed">{stage.desc}</p>
                      <div className="flex justify-between items-center border-t border-white/10 pt-1.5 font-mono text-[9px]">
                        <span className="text-accentCyan">Model: {stage.model}</span>
                        <span className="text-accentPurple-light">Est: {stage.time}</span>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Pipeline Connector */}
              {idx < STAGES.length - 1 && (
                <div className="w-6 lg:w-8 h-1 bg-white/5 rounded-full overflow-hidden shrink-0 relative">
                  {nodeStatus === "completed" && (
                    <div className="absolute inset-0 bg-emerald-500 transition-all duration-500" />
                  )}
                  {nodeStatus === "active" && (
                    <div className="absolute inset-0 bg-gradient-to-r from-emerald-500 to-accentPurple animated-connector animate-pulse" />
                  )}
                  {nodeStatus === "error" && (
                    <div className="absolute inset-0 bg-red-500" />
                  )}
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};
