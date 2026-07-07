import React from "react";
import { motion } from "framer-motion";
import {
  Activity,
  Timer,
  FileVideo,
  Brain,
  ShieldCheck,
  Hash,
  Database,
  Compass,
  CheckCircle2,
} from "lucide-react";

interface MetricsCardProps {
  currentStage: string;
  elapsedTime: number;
  framesProcessed: number;
  currentModel: string;
  confidence: number;
  captionCount: number;
  tokenUsage: number;
  latency: string;
  apiStatus: string;
}

export const MetricsCard: React.FC<MetricsCardProps> = ({
  currentStage,
  elapsedTime,
  framesProcessed,
  currentModel,
  confidence,
  captionCount,
  tokenUsage,
  latency,
  apiStatus
}) => {
  const metrics = [
    {
      id: "stage",
      label: "Current Stage",
      value: currentStage,
      icon: Compass,
      color: "from-accentPurple to-purple-400",
      glow: "shadow-accentPurple/10",
    },
    {
      id: "time",
      label: "Elapsed Time",
      value: `${elapsedTime.toFixed(1)}s`,
      icon: Timer,
      color: "from-accentBlue to-blue-400",
      glow: "shadow-accentBlue/10",
    },
    {
      id: "frames",
      label: "Frames Ingested",
      value: `${framesProcessed} frames`,
      icon: FileVideo,
      color: "from-accentCyan to-cyan-400",
      glow: "shadow-accentCyan/10",
    },
    {
      id: "model",
      label: "Current Model",
      value: currentModel,
      icon: Brain,
      color: "from-accentPurple to-accentBlue",
      glow: "shadow-accentPurple/5",
    },
    {
      id: "confidence",
      label: "Confidence Score",
      value: `${(confidence * 100).toFixed(0)}%`,
      icon: ShieldCheck,
      color: "from-emerald-400 to-teal-500",
      glow: "shadow-emerald-500/10",
    },
    {
      id: "captions",
      label: "Captions Generated",
      value: `${captionCount} styles`,
      icon: Hash,
      color: "from-accentCyan to-accentBlue",
      glow: "shadow-accentCyan/5",
    },
    {
      id: "tokens",
      label: "Token Usage",
      value: tokenUsage.toLocaleString(),
      icon: Database,
      color: "from-accentPurple to-indigo-500",
      glow: "shadow-accentPurple/10",
    },
    {
      id: "latency",
      label: "Overall Latency",
      value: latency,
      icon: Activity,
      color: "from-pink-400 to-accentPurple",
      glow: "shadow-pink-500/10",
    },
    {
      id: "api",
      label: "API Gateway Status",
      value: apiStatus,
      icon: CheckCircle2,
      color: "from-emerald-400 to-green-500",
      glow: "shadow-emerald-500/10",
    },
  ];

  return (
    <div className="w-full grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-9 gap-3">
      {metrics.map((m) => {
        const Icon = m.icon;
        return (
          <motion.div
            key={m.id}
            whileHover={{ y: -4, scale: 1.02 }}
            className={`glass-card p-3 rounded-xl border border-white/5 flex flex-col justify-between shadow-lg ${m.glow} transition-all duration-300 relative overflow-hidden`}
          >
            {/* Shimmer light effect inside */}
            <div className="absolute inset-0 bg-gradient-to-tr from-white/0 via-white/[0.02] to-white/0 pointer-events-none" />

            {/* Label and Icon */}
            <div className="flex items-center justify-between text-gray-500 text-[10px] uppercase font-mono tracking-wider mb-2">
              <span className="truncate pr-1">{m.label}</span>
              <Icon className="w-3.5 h-3.5 shrink-0 text-gray-400" />
            </div>

            {/* Value block */}
            <div className="flex flex-col">
              <span className={`text-sm lg:text-base font-extrabold tracking-tight bg-gradient-to-r ${m.color} bg-clip-text text-transparent truncate`}>
                {m.value}
              </span>
            </div>
          </motion.div>
        );
      })}
    </div>
  );
};
