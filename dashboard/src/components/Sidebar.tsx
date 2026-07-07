import React, { useState } from "react";
import { motion } from "framer-motion";
import { 
  Home, 
  History, 
  Settings, 
  BarChart3, 
  Info, 
  ChevronRight, 
  Cpu 
} from "lucide-react";

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, setActiveTab }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const menuItems = [
    { id: "dashboard", label: "OS Dashboard", icon: Home, desc: "Main control panel" },
    { id: "history", label: "Process History", icon: History, desc: "Past caption logs" },
    { id: "metrics", label: "System Metrics", icon: BarChart3, desc: "Inference latency & cost" },
    { id: "settings", label: "Model Config", icon: Settings, desc: "Adjust pipeline details" },
    { id: "about", label: "Platform Info", icon: Info, desc: "Gemma model overview" },
  ];

  return (
    <motion.div
      className="fixed left-4 top-20 bottom-4 z-50 glass-card rounded-2xl flex flex-col items-center justify-between py-6 border border-white/5 transition-all duration-300"
      animate={{ width: isExpanded ? 240 : 70 }}
      onMouseEnter={() => setIsExpanded(true)}
      onMouseLeave={() => setIsExpanded(false)}
      aria-label="Sidebar Navigation"
    >
      {/* Top Section - Brand/Logo Icon */}
      <div className="w-full px-4 flex items-center justify-start space-x-3 overflow-hidden">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-accentPurple to-accentCyan flex items-center justify-center shadow-lg shadow-accentPurple/20 shrink-0">
          <Cpu className="w-5 h-5 text-white animate-pulse" />
        </div>
        <motion.div
          className="flex flex-col whitespace-nowrap overflow-hidden"
          animate={{ opacity: isExpanded ? 1 : 0, display: isExpanded ? "flex" : "none" }}
        >
          <span className="font-extrabold text-sm tracking-wide bg-gradient-to-r from-white via-gray-100 to-gray-400 bg-clip-text text-transparent">
            ANTIGRAVITY
          </span>
          <span className="text-[10px] text-accentCyan font-mono uppercase tracking-widest">
            Gemma Core v2.0
          </span>
        </motion.div>
      </div>

      {/* Middle Section - Navigation Links */}
      <nav className="flex-1 w-full px-2 py-8 flex flex-col space-y-2 justify-center">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center space-x-3 p-3 rounded-xl transition-all duration-300 relative group overflow-hidden ${
                isActive 
                  ? "bg-accentPurple/10 text-white border border-accentPurple/20" 
                  : "text-gray-400 hover:bg-white/5 hover:text-gray-200 border border-transparent"
              }`}
            >
              {/* Active glow indicator */}
              {isActive && (
                <motion.div
                  layoutId="active-indicator"
                  className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-accentPurple to-accentCyan rounded-r-full"
                />
              )}

              <Icon className={`w-5 h-5 shrink-0 ${isActive ? "text-accentPurple-light" : "group-hover:text-white"}`} />
              
              <motion.div
                className="flex flex-col text-left whitespace-nowrap overflow-hidden"
                animate={{ opacity: isExpanded ? 1 : 0, display: isExpanded ? "flex" : "none" }}
              >
                <span className="text-sm font-semibold">{item.label}</span>
                <span className="text-[10px] text-gray-500 font-medium">{item.desc}</span>
              </motion.div>
            </button>
          );
        })}
      </nav>

      {/* Bottom Section - Expand indicator or Footer */}
      <div className="w-full px-4 flex items-center justify-start shrink-0 overflow-hidden">
        <motion.div
          className="flex items-center text-xs text-gray-500 whitespace-nowrap space-x-2"
          animate={{ opacity: isExpanded ? 1 : 0 }}
        >
          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
          <span>System Online</span>
        </motion.div>
        {!isExpanded && (
          <ChevronRight className="w-4 h-4 text-gray-600 animate-bounce" />
        )}
      </div>
    </motion.div>
  );
};
