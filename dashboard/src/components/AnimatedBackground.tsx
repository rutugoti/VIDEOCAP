import React from "react";
import { motion } from "framer-motion";

export const AnimatedBackground: React.FC = () => {
  return (
    <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
      {/* Glow Blob 1 */}
      <motion.div
        className="absolute w-[600px] h-[600px] rounded-full bg-accentPurple/10 blur-[120px]"
        animate={{
          x: [100, 300, 100],
          y: [100, 200, 100],
          scale: [1, 1.2, 1],
        }}
        transition={{
          duration: 15,
          repeat: Infinity,
          ease: "easeInOut",
        }}
        style={{ left: "-10%", top: "-10%" }}
      />

      {/* Glow Blob 2 */}
      <motion.div
        className="absolute w-[500px] h-[500px] rounded-full bg-accentBlue/10 blur-[100px]"
        animate={{
          x: [200, 50, 200],
          y: [300, 100, 300],
          scale: [1.2, 0.9, 1.2],
        }}
        transition={{
          duration: 12,
          repeat: Infinity,
          ease: "easeInOut",
        }}
        style={{ right: "-5%", bottom: "10%" }}
      />

      {/* Glow Blob 3 */}
      <motion.div
        className="absolute w-[400px] h-[400px] rounded-full bg-accentCyan/8 blur-[90px]"
        animate={{
          x: [-100, 200, -100],
          y: [200, 400, 200],
          scale: [0.8, 1.1, 0.8],
        }}
        transition={{
          duration: 18,
          repeat: Infinity,
          ease: "easeInOut",
        }}
        style={{ left: "30%", top: "40%" }}
      />

      {/* Fine Particle Grid Overlay */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#ffffff03_1px,transparent_1px),linear-gradient(to_bottom,#ffffff03_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_50%,#000_70%,transparent_100%)]" />
    </div>
  );
};
