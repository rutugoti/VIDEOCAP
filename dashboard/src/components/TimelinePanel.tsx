import React from "react";
import { motion } from "framer-motion";
import { Clock, CheckCircle2, XCircle, HelpCircle } from "lucide-react";

interface TimelineEntry {
  stage: string;
  start_time: number;
  end_time: number;
  duration: number;
  model_used: string;
  status: string; // "completed", "failed", "running"
}

interface TimelinePanelProps {
  timeline: TimelineEntry[];
}

export const TimelinePanel: React.FC<TimelinePanelProps> = ({ timeline }) => {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle2 className="w-4 h-4 text-emerald-400" />;
      case "failed":
        return <XCircle className="w-4 h-4 text-red-400" />;
      case "running":
        return <div className="w-2.5 h-2.5 rounded-full bg-accentBlue animate-pulse" />;
      default:
        return <HelpCircle className="w-4 h-4 text-gray-500" />;
    }
  };

  return (
    <div className="glass-card p-4 rounded-xl border border-white/5 flex flex-col space-y-4">
      <div className="flex items-center space-x-2 border-b border-white/5 pb-2">
        <Clock className="w-4 h-4 text-accentPurple" />
        <span className="text-xs font-mono text-gray-400 uppercase tracking-wider">
          Pipeline Execution Timeline (Real Logs)
        </span>
      </div>

      {timeline.length === 0 ? (
        <div className="text-center py-6 text-xs text-gray-500 font-mono">
          NO_EXECUTION_TIMINGS_YET
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left font-mono text-[10px] text-gray-400">
            <thead>
              <tr className="border-b border-white/10 text-gray-500 uppercase tracking-wider text-[8px]">
                <th className="py-2 px-3">Stage</th>
                <th className="py-2 px-3">Start (S)</th>
                <th className="py-2 px-3">End (S)</th>
                <th className="py-2 px-3">Duration (S)</th>
                <th className="py-2 px-3">Engine Model</th>
                <th className="py-2 px-3 text-right">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {timeline.map((entry, idx) => (
                <motion.tr
                  key={idx}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className="hover:bg-white/2 transition-colors"
                >
                  <td className="py-2.5 px-3 font-semibold text-white capitalize">
                    {entry.stage.replace("_", " ")}
                  </td>
                  <td className="py-2.5 px-3">{entry.start_time.toFixed(2)}s</td>
                  <td className="py-2.5 px-3">{entry.end_time.toFixed(2)}s</td>
                  <td className="py-2.5 px-3 text-accentCyan">
                    {entry.duration > 0 ? `${entry.duration.toFixed(2)}s` : "-"}
                  </td>
                  <td className="py-2.5 px-3 text-gray-300 max-w-[120px] truncate" title={entry.model_used}>
                    {entry.model_used}
                  </td>
                  <td className="py-2.5 px-3 flex justify-end items-center space-x-1.5 text-right">
                    {getStatusIcon(entry.status)}
                    <span className={`text-[9px] uppercase font-bold ${
                      entry.status === "completed" 
                        ? "text-emerald-400" 
                        : entry.status === "failed" 
                        ? "text-red-400" 
                        : "text-accentBlue"
                    }`}>
                      {entry.status}
                    </span>
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
