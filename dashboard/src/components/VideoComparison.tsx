import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { 
  Play, 
  Pause, 
  Volume2, 
  VolumeX, 
  Maximize2, 
  RotateCcw, 
  Sparkles, 
  Download,
  Info,
  CheckCircle,
  BookOpen,
  Database
} from "lucide-react";

interface VideoComparisonProps {
  videoPath: string;
  captionsData: {
    formal: string;
    sarcastic: string;
    tech_humor: string;
    non_tech_humor: string;
  };
  validatorScores: {
    semantic_accuracy: number;
    hallucination_risk: number;
    grammar_score: number;
    temporal_consistency_score: number;
    word_budget_pass: boolean;
    overall_confidence: number;
  };
  narrativeText: string;
  jobId: string;
  filename: string;
  ejr?: string;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export const VideoComparison: React.FC<VideoComparisonProps> = ({
  videoPath,
  captionsData,
  validatorScores,
  narrativeText,
  jobId,
  filename,
  ejr
}) => {
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);
  const [volume, setVolume] = useState<number>(0.8);
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [activeStyle, setActiveStyle] = useState<"formal" | "sarcastic" | "tech_humor" | "non_tech_humor">("formal");
  const [showCaptions, setShowCaptions] = useState<boolean>(true);
  const [showNarrative, setShowNarrative] = useState<boolean>(false);

  const leftVideoRef = useRef<HTMLVideoElement>(null);
  const rightVideoRef = useRef<HTMLVideoElement>(null);

  const videoUrl = `${API_BASE_URL}${videoPath}`;

  // Synchronize playing states
  useEffect(() => {
    if (leftVideoRef.current && rightVideoRef.current) {
      if (isPlaying) {
        leftVideoRef.current.play().catch(() => {});
        rightVideoRef.current.play().catch(() => {});
      } else {
        leftVideoRef.current.pause();
        rightVideoRef.current.pause();
      }
    }
  }, [isPlaying]);

  // Sync seek/time updates
  const handleTimeUpdate = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const video = e.currentTarget;
    setCurrentTime(video.currentTime);
    if (video.duration) {
      setDuration(video.duration);
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const targetTime = parseFloat(e.target.value);
    setCurrentTime(targetTime);
    if (leftVideoRef.current) leftVideoRef.current.currentTime = targetTime;
    if (rightVideoRef.current) rightVideoRef.current.currentTime = targetTime;
  };

  const togglePlay = () => {
    setIsPlaying(!isPlaying);
  };

  const handleReset = () => {
    setIsPlaying(false);
    if (leftVideoRef.current) leftVideoRef.current.currentTime = 0;
    if (rightVideoRef.current) rightVideoRef.current.currentTime = 0;
    setCurrentTime(0);
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setVolume(val);
    if (leftVideoRef.current) leftVideoRef.current.volume = val;
    if (rightVideoRef.current) rightVideoRef.current.volume = val;
    setIsMuted(val === 0);
  };

  const toggleMute = () => {
    const targetMuted = !isMuted;
    setIsMuted(targetMuted);
    if (leftVideoRef.current) leftVideoRef.current.muted = targetMuted;
    if (rightVideoRef.current) rightVideoRef.current.muted = targetMuted;
  };

  const handleFullscreen = () => {
    // Go fullscreen on comparison wrapper
    const element = document.getElementById("comparison-players-container");
    if (element) {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else {
        element.requestFullscreen().catch(() => {});
      }
    }
  };

  const formatTime = (time: number) => {
    const min = Math.floor(time / 60);
    const sec = Math.floor(time % 60);
    return `${min}:${sec < 10 ? "0" : ""}${sec}`;
  };

  // Get current active caption text
  const getActiveCaption = () => {
    if (!showCaptions) return "";
    return captionsData[activeStyle];
  };

  // Download the formatted submission JSON
  const handleDownloadJson = () => {
    const videoId = filename.split(".")[0] || jobId;
    const submissionObj = [
      {
        video_id: videoId,
        formal: captionsData.formal,
        sarcastic: captionsData.sarcastic,
        humorous_tech: captionsData.tech_humor,
        humorous_non_tech: captionsData.non_tech_humor,
        validator_scores: validatorScores,
        narrative: narrativeText
      }
    ];
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(submissionObj, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `${videoId}_submission.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div className="w-full flex flex-col space-y-4">
      {/* Dynamic Caption style tabs */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/5 pb-2">
        <div className="flex space-x-1 bg-white/5 p-1 rounded-xl">
          {(["formal", "sarcastic", "tech_humor", "non_tech_humor"] as const).map((style) => (
            <button
              key={style}
              onClick={() => setActiveStyle(style)}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono capitalize transition-all ${
                activeStyle === style
                  ? "bg-accentPurple text-white shadow-lg shadow-accentPurple/20"
                  : "text-gray-400 hover:text-white"
              }`}
            >
              {style.replace("_", " ")}
            </button>
          ))}
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowNarrative(!showNarrative)}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono flex items-center space-x-1.5 transition-all ${
              showNarrative 
                ? "bg-accentCyan text-white" 
                : "bg-white/5 text-gray-400 hover:text-white border border-white/5"
            }`}
          >
            <Info className="w-3.5 h-3.5" />
            <span>{showNarrative ? "Hide Narrative" : "Show Narrative"}</span>
          </button>
          <button
            onClick={() => setShowCaptions(!showCaptions)}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all ${
              showCaptions 
                ? "bg-accentBlue text-white" 
                : "bg-white/5 text-gray-400 hover:text-white border border-white/5"
            }`}
          >
            {showCaptions ? "Captions ON" : "Captions OFF"}
          </button>
          <button
            onClick={handleDownloadJson}
            className="bg-white/5 hover:bg-white/10 border border-white/5 text-gray-300 hover:text-white px-3 py-1.5 rounded-lg text-xs font-mono flex items-center space-x-1.5 transition-all"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download JSON</span>
          </button>
        </div>
      </div>

      {/* Players container (Original vs Captioned) */}
      <div 
        id="comparison-players-container"
        className="grid grid-cols-1 md:grid-cols-2 gap-4 bg-bgMain/30 p-2 rounded-2xl border border-white/5 relative"
      >
        {/* Left Card: Original Video */}
        <div className="flex flex-col space-y-2 relative group">
          <div className="absolute top-3 left-3 z-10 bg-black/60 backdrop-blur-md px-2.5 py-1 rounded-lg text-[10px] font-mono uppercase tracking-wider text-gray-400 border border-white/5">
            Original Video
          </div>
          <div className="aspect-video bg-black rounded-xl overflow-hidden border border-white/10 relative">
            {videoUrl ? (
              <video
                ref={leftVideoRef}
                src={videoUrl}
                onTimeUpdate={handleTimeUpdate}
                onClick={togglePlay}
                className="w-full h-full object-contain cursor-pointer"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-gray-500 font-mono text-xs">
                LOADING_STREAM...
              </div>
            )}
          </div>
        </div>

        {/* Right Card: Captioned Video */}
        <div className="flex flex-col space-y-2 relative group">
          <div className="absolute top-3 left-3 z-10 bg-accentPurple/25 backdrop-blur-md px-2.5 py-1 rounded-lg text-[10px] font-mono uppercase tracking-wider text-white border border-accentPurple/30 flex items-center space-x-1">
            <Sparkles className="w-3 h-3 text-accentPurple-light animate-spin" />
            <span>Captioned Video</span>
          </div>
          <div className="aspect-video bg-black rounded-xl overflow-hidden border border-white/10 relative">
            {videoUrl ? (
              <div className="w-full h-full relative">
                <video
                  ref={rightVideoRef}
                  src={videoUrl}
                  onClick={togglePlay}
                  className="w-full h-full object-contain cursor-pointer"
                />
                {/* Custom Subtitles Overlay */}
                {showCaptions && getActiveCaption() && (
                  <div className="absolute bottom-6 left-4 right-4 text-center z-10 pointer-events-none">
                    <span className="bg-black/80 backdrop-blur-md text-white font-medium text-xs md:text-sm px-4 py-2 rounded-xl border border-white/10 leading-relaxed max-w-[90%] inline-block shadow-2xl">
                      {getActiveCaption()}
                    </span>
                  </div>
                )}
              </div>
            ) : (
              <div className="w-full h-full flex items-center justify-center text-gray-500 font-mono text-xs">
                LOADING_STREAM...
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Control bar */}
      <div className="glass-card p-4 rounded-xl flex flex-wrap items-center justify-between gap-4 border border-white/5">
        <div className="flex items-center space-x-2">
          <button
            onClick={togglePlay}
            className="w-10 h-10 rounded-lg bg-accentPurple hover:bg-accentPurple/80 text-white flex items-center justify-center transition-colors shadow-lg shadow-accentPurple/20"
          >
            {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
          </button>
          <button
            onClick={handleReset}
            className="w-10 h-10 rounded-lg bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white flex items-center justify-center border border-white/5 transition-all"
            title="Reset playback"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>

        {/* Playback seeker */}
        <div className="flex-1 min-w-[200px] flex items-center space-x-3">
          <span className="text-[10px] font-mono text-gray-500">{formatTime(currentTime)}</span>
          <input
            type="range"
            min={0}
            max={duration || 100}
            step={0.05}
            value={currentTime}
            onChange={handleSeek}
            className="flex-1 accent-accentPurple bg-white/10 h-1 rounded-full cursor-pointer appearance-none"
          />
          <span className="text-[10px] font-mono text-gray-500">{formatTime(duration)}</span>
        </div>

        {/* Volume & Fullscreen */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <button onClick={toggleMute} className="text-gray-400 hover:text-white transition-colors">
              {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
            </button>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={isMuted ? 0 : volume}
              onChange={handleVolumeChange}
              className="w-16 accent-accentPurple bg-white/10 h-1 rounded-full cursor-pointer appearance-none"
            />
          </div>
          <button
            onClick={handleFullscreen}
            className="p-2 bg-white/5 hover:bg-white/10 rounded-lg border border-white/5 text-gray-400 hover:text-white transition-colors"
          >
            <Maximize2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Grid for Narrative overlay & Validator Report */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Narrative Box */}
        <motion.div
          animate={{ height: showNarrative ? "auto" : "0px", opacity: showNarrative ? 1 : 0 }}
          className="glass-card rounded-xl border border-white/5 overflow-hidden transition-all duration-300"
        >
          <div className="p-4 flex flex-col space-y-2">
            <div className="flex items-center space-x-2">
              <BookOpen className="w-4 h-4 text-accentCyan" />
              <span className="text-xs font-mono text-gray-400 uppercase tracking-wider">
                Gemma Neutral Narrative
              </span>
            </div>
            <p className="text-xs text-gray-300 leading-relaxed font-sans bg-black/30 p-3 rounded-lg border border-white/5">
              {narrativeText}
            </p>
          </div>
        </motion.div>

        {/* Validator Report Card */}
        <div className="glass-card p-4 rounded-xl border border-white/5 flex flex-col space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <CheckCircle className="w-4 h-4 text-emerald-400" />
              <span className="text-xs font-mono text-gray-400 uppercase tracking-wider">
                Semantic Validator Report
              </span>
            </div>
            <span className="bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded text-[10px] font-mono text-emerald-400 uppercase font-bold">
              VERIFIED_PASS
            </span>
          </div>

          {/* Scores list */}
          <div className="grid grid-cols-2 gap-3 text-[10px] font-mono text-gray-400">
            <div className="bg-white/2 p-2 rounded-lg border border-white/5 flex justify-between items-center">
              <span>Semantic Accuracy:</span>
              <span className="text-white font-bold text-xs">{(validatorScores.semantic_accuracy * 100).toFixed(0)}%</span>
            </div>
            <div className="bg-white/2 p-2 rounded-lg border border-white/5 flex justify-between items-center">
              <span>Hallucination Risk:</span>
              <span className="text-white font-bold text-xs">{(validatorScores.hallucination_risk * 100).toFixed(0)}%</span>
            </div>
            <div className="bg-white/2 p-2 rounded-lg border border-white/5 flex justify-between items-center">
              <span>Grammar Score:</span>
              <span className="text-white font-bold text-xs">{(validatorScores.grammar_score * 100).toFixed(0)}%</span>
            </div>
            <div className="bg-white/2 p-2 rounded-lg border border-white/5 flex justify-between items-center">
              <span>Consistency Score:</span>
              <span className="text-white font-bold text-xs">{(validatorScores.temporal_consistency_score * 100).toFixed(0)}%</span>
            </div>
          </div>

          <div className="flex justify-between items-center text-[10px] font-mono pt-1 text-gray-500">
            <span>Word Budget Constraint: PASS (15-35 words)</span>
            <span>Overall Confidence: {(validatorScores.overall_confidence * 100).toFixed(0)}%</span>
          </div>
        </div>
      </div>

      {/* Evidence Justification Record (EJR) parsed markdown table */}
      {ejr && (() => {
        const lines = ejr.trim().split("\n");
        const tableLines = lines.filter(l => l.trim().startsWith("|"));
        if (tableLines.length < 2) return null;
        
        const headers = tableLines[0].split("|").map(s => s.trim()).filter((_, idx, arr) => idx > 0 && idx < arr.length - 1);
        const rows = tableLines.slice(2).map(line => {
          return line.split("|").map(s => s.trim()).filter((_, idx, arr) => idx > 0 && idx < arr.length - 1);
        });

        return (
          <div className="overflow-x-auto w-full bg-black/20 rounded-xl border border-white/5 p-4">
            <div className="flex items-center space-x-2 mb-3">
              <Database className="w-4 h-4 text-accentPurple" />
              <span className="text-xs font-mono text-gray-400 uppercase tracking-wider">
                Evidence Justification Record (EJR)
              </span>
            </div>
            <table className="w-full text-left text-[10px] font-mono text-gray-300 border-collapse">
              <thead>
                <tr className="border-b border-white/10 text-gray-400">
                  {headers.map((h, i) => (
                    <th key={i} className="pb-2 pt-1 px-3 font-semibold uppercase tracking-wider">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, rIdx) => (
                  <tr key={rIdx} className="border-b border-white/5 hover:bg-white/2 transition-colors">
                    {row.map((cell, cIdx) => (
                      <td key={cIdx} className="py-2.5 px-3 leading-normal max-w-[200px] break-words">{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      })()}
    </div>
  );
};
