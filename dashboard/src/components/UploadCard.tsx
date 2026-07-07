import React, { useState, useRef } from "react";
import { motion } from "framer-motion";
import { UploadCloud, Film, AlertTriangle, Loader2 } from "lucide-react";

interface UploadCardProps {
  onUploadSuccess: (jobId: string) => void;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export const UploadCard: React.FC<UploadCardProps> = ({ onUploadSuccess }) => {
  const [isDragActive, setIsDragActive] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    if (isUploading) return;
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const uploadFile = async (file: File) => {
    setIsUploading(true);
    setErrorMsg(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || "Failed to upload video to backend.");
      }

      const data = await response.json();
      if (data.job_id) {
        onUploadSuccess(data.job_id);
      } else {
        throw new Error("Invalid response from server (missing job_id).");
      }
    } catch (err: any) {
      setErrorMsg(err.message || "Network error uploading video. Check backend status.");
    } finally {
      setIsUploading(false);
    }
  };

  const validateAndProcessFile = (file: File) => {
    setErrorMsg(null);
    const ext = file.name.split(".").pop()?.toLowerCase();
    const validExtensions = ["mp4", "mov", "avi", "mkv"];

    if (!ext || !validExtensions.includes(ext)) {
      setErrorMsg("Unsupported file format. Please upload MP4, MOV, AVI, or MKV.");
      return;
    }

    if (file.size > 100 * 1024 * 1024) {
      setErrorMsg("Video size exceeds the 100MB limit.");
      return;
    }

    uploadFile(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    if (isUploading) return;
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndProcessFile(e.dataTransfer.files[0]);
    }
  };

  const handleBrowseClick = () => {
    if (isUploading) return;
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndProcessFile(e.target.files[0]);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.5 }}
      className="w-full h-full flex items-center justify-center p-4"
    >
      <div className="w-full max-w-4xl h-[450px] glass-card rounded-3xl p-6 relative overflow-hidden flex flex-col justify-between border border-white/5 shadow-2xl">
        
        {/* Glow corner elements */}
        <div className="absolute -top-16 -left-16 w-32 h-32 rounded-full bg-accentPurple/20 blur-xl pointer-events-none" />
        <div className="absolute -bottom-16 -right-16 w-32 h-32 rounded-full bg-accentCyan/20 blur-xl pointer-events-none" />

        {/* Top Header info */}
        <div className="flex justify-between items-center z-10">
          <div className="flex items-center space-x-2">
            <Film className="w-4 h-4 text-accentCyan" />
            <span className="text-xs font-mono tracking-wider text-gray-400 uppercase">
              Video Loader Ingestion Layer
            </span>
          </div>
          <span className="text-[10px] text-gray-500 font-mono">
            MAX_DURATION: 120S
          </span>
        </div>

        {/* Upload Drop Zone */}
        <div
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={handleBrowseClick}
          className={`flex-1 mx-4 my-6 rounded-2xl border-2 border-dashed transition-all duration-300 flex flex-col items-center justify-center cursor-pointer group relative overflow-hidden ${
            isDragActive 
              ? "border-accentPurple bg-accentPurple/5" 
              : "border-white/10 hover:border-accentCyan/50 hover:bg-white/2"
          } ${isUploading ? "pointer-events-none opacity-60" : ""}`}
        >
          {/* Subtle particle background grid inside zone */}
          <div className="absolute inset-0 bg-[radial-gradient(#ffffff03_1px,transparent_1px)] bg-[size:16px_16px] pointer-events-none" />

          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept=".mp4,.mov,.avi,.mkv"
            disabled={isUploading}
            onChange={handleFileChange}
          />

          <motion.div 
            animate={{ y: isDragActive ? -10 : 0 }}
            className={`w-16 h-16 rounded-full bg-white/5 border border-white/10 flex items-center justify-center mb-4 transition-colors group-hover:bg-accentCyan/10 group-hover:border-accentCyan/30 ${
              isDragActive ? "bg-accentPurple/10 border-accentPurple/30" : ""
            }`}
          >
            {isUploading ? (
              <Loader2 className="w-8 h-8 text-accentPurple animate-spin" />
            ) : (
              <UploadCloud className={`w-8 h-8 transition-colors ${
                isDragActive ? "text-accentPurple-light" : "text-gray-400 group-hover:text-accentCyan-light"
              }`} />
            )}
          </motion.div>

          <h3 className="text-lg font-bold text-white tracking-wide mb-1 group-hover:text-accentCyan-light transition-colors">
            {isUploading ? "Uploading file to server..." : "Drop your video here"}
          </h3>
          <p className="text-xs text-gray-500 mb-4">
            {isUploading ? "Please do not close this window" : (
              <>or <span className="text-accentPurple-light font-semibold group-hover:underline">Browse Files</span></>
            )}
          </p>

          <div className="flex space-x-6 text-[10px] text-gray-600 font-mono">
            <span>SUPPORTED: MP4, MOV, AVI, MKV</span>
            <span>•</span>
            <span>MAX: 100MB</span>
          </div>
        </div>

        {/* Footer/Errors */}
        <div className="h-10 flex items-center justify-center z-10">
          {errorMsg ? (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center space-x-2 text-red-400 text-xs bg-red-500/10 border border-red-500/20 px-4 py-1.5 rounded-xl"
            >
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>{errorMsg}</span>
            </motion.div>
          ) : (
            <span className="text-[10px] text-gray-500 text-center font-mono">
              SECURE_SANDBOX: ALL INFERENCE RUNS LOCALLY IN COMPLIANCE WITH RULESETS
            </span>
          )}
        </div>
      </div>
    </motion.div>
  );
};
