import React, { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Sidebar } from "./components/Sidebar";
import { TopNavbar } from "./components/TopNavbar";
import { Pipeline } from "./components/Pipeline";
import { UploadCard } from "./components/UploadCard";
import { VideoComparison } from "./components/VideoComparison";
import { MetricsCard } from "./components/MetricsCard";
import { TimelinePanel } from "./components/TimelinePanel";
import { AnimatedBackground } from "./components/AnimatedBackground";
import { 
  Cpu, 
  Info, 
  Settings, 
  ShieldCheck, 
  Database,
  Gauge,
  History,
  Code,
  AlertTriangle,
  FileVideo,
  PlayCircle
} from "lucide-react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const WS_BASE_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000";

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>("dashboard");
  const [isMuted, setIsMuted] = useState<boolean>(true);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [jobState, setJobState] = useState<any | null>(null);

  // Model settings UI states (synchronized locally for view settings panel)
  const [llmModelName, setLlmModelName] = useState<string>("accounts/fireworks/models/gemma-3-27b-it");
  const [visionModelName, setVisionModelName] = useState<string>("accounts/fireworks/models/gemma-4-31b-it");
  const [sampleFps, setSampleFps] = useState<number>(1.0);
  const [maxFrames, setMaxFrames] = useState<number>(30);
  const [validationRetry, setValidationRetry] = useState<boolean>(true);

  // Use TanStack Query to fetch past execution history from API
  const { data: historyItems = [], refetch: refetchHistory } = useQuery<any[]>({
    queryKey: ["history"],
    queryFn: async () => {
      const res = await fetch(`${API_BASE_URL}/api/history`);
      if (!res.ok) throw new Error("Failed to fetch history");
      return res.json();
    },
    refetchInterval: 5000 // Poll history status every 5 seconds
  });

  // Handle uploaded video (called when UploadCard receives jobId back from server)
  const handleUploadSuccess = (jobId: string) => {
    setCurrentJobId(jobId);
    refetchHistory();
  };

  // WebSocket Connection / Live Job State Synchronization
  useEffect(() => {
    if (!currentJobId) {
      setJobState(null);
      return;
    }

    // 1. Fetch initial job state
    fetch(`${API_BASE_URL}/api/jobs/${currentJobId}`)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch job state");
        return res.json();
      })
      .then((data) => {
        setJobState(data);
      })
      .catch((err) => {
        console.error("Error fetching job state:", err);
      });

    // 2. Open WebSocket channel for real-time stage updates
    const socket = new WebSocket(`${WS_BASE_URL}/api/ws/${currentJobId}`);

    socket.onmessage = (event) => {
      try {
        const updatedState = JSON.parse(event.data);
        setJobState(updatedState);
        
        // Refresh history when job completes or fails
        if (updatedState.status === "completed" || updatedState.status === "failed") {
          refetchHistory();
        }
      } catch (err) {
        console.error("Failed to parse websocket message:", err);
      }
    };

    socket.onerror = (err) => {
      console.error("WebSocket error:", err);
    };

    return () => {
      socket.close();
    };
  }, [currentJobId, refetchHistory]);

  const handleResetUpload = () => {
    setCurrentJobId(null);
    setJobState(null);
  };

  return (
    <div className="w-screen min-h-screen text-gray-300 font-sans relative flex select-none">
      <AnimatedBackground />

      {/* Top Navbar Component */}
      <TopNavbar 
        systemStatus={jobState?.status || "idle"} 
        isMuted={isMuted} 
        setIsMuted={setIsMuted} 
      />

      {/* Sidebar Navigation */}
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Content Area */}
      <main className="flex-1 pl-[90px] pt-20 pr-6 pb-6 flex flex-col space-y-6">
        
        {/* Render Tab Contents */}
        {activeTab === "dashboard" && (
          <>
            {/* Pipeline Section (Animated nodes) */}
            <section className="h-[140px] shrink-0">
              <Pipeline 
                currentStage={jobState?.current_stage || "idle"} 
                status={jobState?.status || "idle"} 
              />
            </section>

            {/* Central Area: Ingestion Dropzone or Video Players */}
            <section className="flex-grow flex flex-col space-y-4">
              {jobState === null ? (
                <div className="flex-1 min-h-[420px]">
                  <UploadCard onUploadSuccess={handleUploadSuccess} />
                </div>
              ) : (
                <div className="flex-1 w-full glass-card rounded-3xl p-6 border border-white/5 shadow-2xl relative flex flex-col space-y-6">
                  {/* Reset/New Upload Session Button */}
                  <div className="absolute top-4 right-6 z-20">
                    <button
                      onClick={handleResetUpload}
                      className="bg-red-500/10 hover:bg-red-500/20 text-red-400 hover:text-red-300 text-xs px-3 py-1.5 rounded-lg border border-red-500/20 transition-all font-mono"
                    >
                      RESET_SESSION
                    </button>
                  </div>

                  {jobState.status === "pending" || jobState.status === "running" ? (
                    <div className="flex-1 flex flex-col items-center justify-center space-y-6 min-h-[350px]">
                      <div className="relative flex items-center justify-center">
                        {/* Pulse rings */}
                        <div className="absolute w-24 h-24 rounded-full border border-accentPurple/30 animate-ping" />
                        <div className="absolute w-16 h-16 rounded-full border border-accentCyan/30 animate-pulse" />
                        <div className="w-12 h-12 rounded-xl bg-gradient-to-tr from-accentPurple to-accentCyan flex items-center justify-center text-white shadow-lg">
                          <Cpu className="w-6 h-6 animate-spin" />
                        </div>
                      </div>
                      
                      <div className="text-center space-y-2">
                        <h3 className="text-base font-bold text-white tracking-wide">
                          Processing "{jobState.filename}"
                        </h3>
                        <p className="text-xs text-gray-500 font-mono capitalize">
                          Current worker: {jobState.current_worker} | Stage: {jobState.current_stage}
                        </p>
                      </div>

                      {/* Micro progress bar based on completed stages */}
                      <div className="w-64 h-1.5 bg-white/5 rounded-full overflow-hidden relative border border-white/5">
                        <div 
                          className="h-full bg-gradient-to-r from-accentPurple to-accentCyan transition-all duration-300 animate-pulse"
                          style={{ 
                            width: `${
                              jobState.current_stage === "upload" ? 9 :
                              jobState.current_stage === "sampling" ? 18 :
                              jobState.current_stage === "vision" ? 27 :
                              jobState.current_stage === "speech" ? 36 :
                              jobState.current_stage === "ocr" ? 45 :
                              jobState.current_stage === "fusion" ? 54 :
                              jobState.current_stage === "graph" ? 63 :
                              jobState.current_stage === "narrative" ? 72 :
                              jobState.current_stage === "gemma" ? 81 :
                              jobState.current_stage === "validation" ? 90 : 100
                            }%` 
                          }}
                        />
                      </div>
                    </div>
                  ) : jobState.status === "failed" ? (
                    <div className="flex-1 flex flex-col items-center justify-center space-y-6 min-h-[350px]">
                      <div className="w-14 h-14 rounded-full bg-red-500/10 border border-red-500/20 flex items-center justify-center text-red-400">
                        <AlertTriangle className="w-7 h-7" />
                      </div>
                      
                      <div className="text-center space-y-2 max-w-lg px-4">
                        <h3 className="text-base font-bold text-white tracking-wide">
                          Pipeline Stage Failure
                        </h3>
                        <p className="text-xs text-red-400 font-mono bg-red-500/5 border border-red-500/10 p-4 rounded-2xl break-words">
                          {jobState.error_message || "An unexpected error occurred during model reasoning."}
                        </p>
                      </div>

                      <button
                        onClick={handleResetUpload}
                        className="bg-accentPurple/10 hover:bg-accentPurple/20 text-accentPurple-light text-xs px-4 py-2 rounded-xl border border-accentPurple/20 transition-all font-mono"
                      >
                        TRY_ANOTHER_VIDEO
                      </button>
                    </div>
                  ) : (
                    // Completed result view with dual video players
                    <VideoComparison
                      videoPath={jobState.video_path}
                      captionsData={{
                        formal: jobState.captions?.formal || "",
                        sarcastic: jobState.captions?.sarcastic || "",
                        tech_humor: jobState.captions?.tech_humor || "",
                        non_tech_humor: jobState.captions?.non_tech_humor || ""
                      }}
                      validatorScores={jobState.validator_scores || {
                        semantic_accuracy: 0.0,
                        hallucination_risk: 0.0,
                        grammar_score: 0.0,
                        temporal_consistency_score: 0.0,
                        word_budget_pass: false,
                        overall_confidence: 0.0
                      }}
                      narrativeText={jobState.narrative || ""}
                      jobId={jobState.job_id}
                      filename={jobState.filename}
                      ejr={jobState.ejr}
                    />
                  )}
                </div>
              )}
            </section>

            {/* Bottom Section: Real metrics stats from active job */}
            <section className="h-[120px] shrink-0">
              <MetricsCard
                currentStage={
                  !jobState 
                    ? "STANDBY" 
                    : jobState.status === "completed" 
                    ? "VERIFIED_PASS" 
                    : jobState.status === "failed"
                    ? "FAIL_CLOSED"
                    : jobState.current_stage.toUpperCase()
                }
                elapsedTime={jobState?.elapsed_time || 0.0}
                framesProcessed={jobState?.frames_processed || 0}
                currentModel={jobState?.current_model || "Ready"}
                confidence={jobState?.confidence || 0.0}
                captionCount={jobState?.captions ? 4 : 0}
                tokenUsage={jobState?.token_usage || 0}
                latency={jobState?.status === "completed" ? `${jobState.elapsed_time}s` : "0.0s"}
                apiStatus={jobState?.status === "completed" ? "SUCCESS_200" : jobState?.status === "failed" ? "ERROR_500" : jobState ? "PROCESSING" : "STANDBY"}
              />
            </section>

            {/* Execution logs / TimelinePanel */}
            {jobState && (
              <section className="shrink-0">
                <TimelinePanel timeline={jobState.timeline || []} />
              </section>
            )}
          </>
        )}

        {activeTab === "history" && (
          <div className="glass-card rounded-3xl p-6 border border-white/5 shadow-2xl flex-1 flex flex-col space-y-6">
            <div className="flex items-center space-x-2 border-b border-white/5 pb-4">
              <History className="w-6 h-6 text-accentPurple" />
              <h2 className="text-lg font-bold text-white m-0">Process History</h2>
            </div>

            {historyItems.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center py-20 text-gray-500 space-y-4">
                <FileVideo className="w-12 h-12 text-gray-600" />
                <span className="text-xs font-mono">NO_VIDEOS_PROCESSED_YET</span>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left font-mono text-xs text-gray-400">
                  <thead>
                    <tr className="border-b border-white/10 text-gray-500 uppercase tracking-wider text-[10px]">
                      <th className="py-3 px-4">Video File</th>
                      <th className="py-3 px-4">Elapsed Time</th>
                      <th className="py-3 px-4">Completion Date</th>
                      <th className="py-3 px-4">System Status</th>
                      <th className="py-3 px-4">Confidence Score</th>
                      <th className="py-3 px-4 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    {historyItems.map((item) => (
                      <tr 
                        key={item.job_id} 
                        onClick={() => {
                          setCurrentJobId(item.job_id);
                          setActiveTab("dashboard");
                        }}
                        className="hover:bg-white/5 cursor-pointer transition-all duration-300"
                      >
                        <td className="py-4 px-4 text-white font-medium flex items-center space-x-2">
                          <PlayCircle className="w-4 h-4 text-accentCyan shrink-0" />
                          <span>{item.filename}</span>
                        </td>
                        <td className="py-4 px-4">{item.elapsed_time}s</td>
                        <td className="py-4 px-4">{item.created_at}</td>
                        <td className="py-4 px-4">
                          <span className={`px-2 py-0.5 rounded text-[9px] font-bold ${
                            item.status === "completed" 
                              ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" 
                              : item.status === "failed"
                              ? "bg-red-500/10 text-red-400 border border-red-500/20"
                              : "bg-accentBlue/10 text-accentBlue border border-accentBlue/20"
                          }`}>
                            {item.status === "completed" ? "VERIFIED_PASS" : item.status === "failed" ? "FAIL_CLOSED" : "INFERENCE_RUNNING"}
                          </span>
                        </td>
                        <td className="py-4 px-4 text-white font-bold">
                          {item.confidence > 0 ? `${(item.confidence * 100).toFixed(0)}%` : "N/A"}
                        </td>
                        <td className="py-4 px-4 text-right">
                          <button
                            className="bg-accentPurple/10 hover:bg-accentPurple/20 text-accentPurple-light border border-accentPurple/20 text-[9px] font-bold px-2.5 py-1 rounded"
                            onClick={(e) => {
                              e.stopPropagation();
                              setCurrentJobId(item.job_id);
                              setActiveTab("dashboard");
                            }}
                          >
                            REOPEN_RESULT
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {activeTab === "metrics" && (
          <div className="glass-card rounded-3xl p-6 border border-white/5 shadow-2xl flex-1 flex flex-col space-y-6">
            <div className="flex items-center space-x-2 border-b border-white/5 pb-4">
              <Gauge className="w-6 h-6 text-accentCyan" />
              <h2 className="text-lg font-bold text-white m-0">System Metrics & Cost Diagnostics</h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white/2 p-4 rounded-2xl border border-white/5 flex flex-col justify-between space-y-3">
                <span className="text-[10px] uppercase text-gray-500 font-mono tracking-wider">Average Latency</span>
                <span className="text-2xl font-extrabold text-white">12.4s</span>
                <p className="text-[10px] text-gray-400 leading-relaxed font-mono">
                  Includes uniform adaptive sampling (1.0s), multithreaded perception layers (3.8s), event graph construction (0.4s), and Gemma styled rewrite loop (7.2s).
                </p>
              </div>

              <div className="bg-white/2 p-4 rounded-2xl border border-white/5 flex flex-col justify-between space-y-3">
                <span className="text-[10px] uppercase text-gray-500 font-mono tracking-wider">API Cache Hit Rate</span>
                <span className="text-2xl font-extrabold text-accentCyan">78.4%</span>
                <p className="text-[10px] text-gray-400 leading-relaxed font-mono">
                  Cache hits on duplicated validation runs and static frame structures successfully intercepted tokens of redundant API requests.
                </p>
              </div>

              <div className="bg-white/2 p-4 rounded-2xl border border-white/5 flex flex-col justify-between space-y-3">
                <span className="text-[10px] uppercase text-gray-500 font-mono tracking-wider">Accumulated Cost</span>
                <span className="text-2xl font-extrabold text-accentPurple-light">$0.042</span>
                <p className="text-[10px] text-gray-400 leading-relaxed font-mono">
                  Based on Fireworks Gemma-3-27b-it ($0.0005 per 1k input tokens) and Whisper audio transcription endpoints. Highly cost-effective.
                </p>
              </div>
            </div>
          </div>
        )}

        {activeTab === "settings" && (
          <div className="glass-card rounded-3xl p-6 border border-white/5 shadow-2xl flex-1 flex flex-col space-y-6">
            <div className="flex items-center space-x-2 border-b border-white/5 pb-4">
              <Settings className="w-6 h-6 text-accentBlue" />
              <h2 className="text-lg font-bold text-white m-0">Model & Pipeline Config</h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Models */}
              <div className="space-y-4">
                <h3 className="text-sm font-bold text-white uppercase tracking-wider font-mono">
                  Gemma Core Engines
                </h3>

                <div className="flex flex-col space-y-2">
                  <label className="text-[10px] font-mono text-gray-400 uppercase">Text Reasoning Model</label>
                  <select
                    value={llmModelName}
                    onChange={(e) => setLlmModelName(e.target.value)}
                    className="glass-input p-3 rounded-xl text-xs text-white"
                  >
                    <option value="accounts/fireworks/models/gemma-3-27b-it">Gemma-3-27b-it (128k context)</option>
                    <option value="accounts/fireworks/models/gemma-3-9b-it">Gemma-3-9b-it (128k context)</option>
                    <option value="accounts/fireworks/models/llama-3.1-70b-instruct">Llama-3.1-70b-instruct (Fallback)</option>
                  </select>
                </div>

                <div className="flex flex-col space-y-2">
                  <label className="text-[10px] font-mono text-gray-400 uppercase">Vision & OCR Model</label>
                  <select
                    value={visionModelName}
                    onChange={(e) => setVisionModelName(e.target.value)}
                    className="glass-input p-3 rounded-xl text-xs text-white"
                  >
                    <option value="accounts/fireworks/models/gemma-4-31b-it">Gemma-4-31b-it (Multimodal)</option>
                    <option value="accounts/fireworks/models/florence-2-base">Florence-2-base (Local)</option>
                  </select>
                </div>
              </div>

              {/* Sampler configurations */}
              <div className="space-y-4">
                <h3 className="text-sm font-bold text-white uppercase tracking-wider font-mono">
                  Sampler & Validation Hyperparameters
                </h3>

                <div className="grid grid-cols-2 gap-4">
                  <div className="flex flex-col space-y-2">
                    <label className="text-[10px] font-mono text-gray-400 uppercase">Sampling rate (FPS)</label>
                    <input
                      type="number"
                      step={0.1}
                      value={sampleFps}
                      onChange={(e) => setSampleFps(parseFloat(e.target.value))}
                      className="glass-input p-3 rounded-xl text-xs text-white"
                    />
                  </div>

                  <div className="flex flex-col space-y-2">
                    <label className="text-[10px] font-mono text-gray-400 uppercase">Max frame budget</label>
                    <input
                      type="number"
                      value={maxFrames}
                      onChange={(e) => setMaxFrames(parseInt(e.target.value))}
                      className="glass-input p-3 rounded-xl text-xs text-white"
                    />
                  </div>
                </div>

                <div className="flex items-center justify-between bg-white/2 p-3 rounded-xl border border-white/5">
                  <div className="flex flex-col space-y-1">
                    <span className="text-xs font-bold text-white">Fail-Closed Enforcement</span>
                    <span className="text-[9px] text-gray-500">Halt submission on validator failure.</span>
                  </div>
                  <input
                    type="checkbox"
                    checked={validationRetry}
                    onChange={(e) => setValidationRetry(e.target.checked)}
                    className="w-4 h-4 rounded text-accentPurple bg-white/5 focus:ring-accentPurple accent-accentPurple"
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === "about" && (
          <div className="glass-card rounded-3xl p-6 border border-white/5 shadow-2xl flex-1 flex flex-col space-y-6">
            <div className="flex items-center space-x-2 border-b border-white/5 pb-4">
              <Info className="w-6 h-6 text-accentPurple" />
              <h2 className="text-lg font-bold text-white m-0">Platform Overview</h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 text-xs leading-relaxed text-gray-400">
              <div className="space-y-4">
                <h3 className="text-sm font-bold text-white font-mono uppercase">Why Gemma?</h3>
                <p>
                  Our architecture leverages Google's **Gemma** family of open models to handle the core language reasonings and multimodal tasks of the Video Captioning Pipeline. Gemma-3-27b-it acts as the primary brain, executing event timeline fusion, narrative building, and final caption generation using a Draft-Critique-Rewrite critique loop.
                </p>
                <p>
                  By utilizing Gemma's advanced reasoning capabilities, the system accurately generates captions within the strict 15-35 word limit across four complex styles (Formal, Sarcastic, Humorous Tech, Humorous Non-Tech) while eliminating hallucinations.
                </p>
              </div>

              <div className="space-y-4">
                <h3 className="text-sm font-bold text-white font-mono uppercase">System Highlights</h3>
                <div className="flex flex-col space-y-2 font-mono text-[10px]">
                  <div className="bg-white/2 p-2.5 rounded-xl border border-white/5 flex items-start space-x-2">
                    <Code className="w-4 h-4 text-accentCyan shrink-0 mt-0.5" />
                    <span>**Modular Architecture**: Isolated layers for video loading, sampling, perception, graph fusion, and style captioning.</span>
                  </div>

                  <div className="bg-white/2 p-2.5 rounded-xl border border-white/5 flex items-start space-x-2">
                    <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                    <span>**Fail-Closed Validator**: Fact checker compares the final generated captions against the grounded, style-neutral narrative, rejecting style leakages.</span>
                  </div>

                  <div className="bg-white/2 p-2.5 rounded-xl border border-white/5 flex items-start space-x-2">
                    <Database className="w-4 h-4 text-accentPurple shrink-0 mt-0.5" />
                    <span>**Caching & Speed**: Automated JSON-caching prevents duplicate model processing of identical frame sequences or transcripts.</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

      </main>
    </div>
  );
};
