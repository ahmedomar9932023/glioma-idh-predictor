"use client";

import { useEffect, useRef } from "react";

interface AnimatedCandleProps {
  size?: "sm" | "md" | "lg" | "xl";
  glowing?: boolean;
  className?: string;
}

const sizes = {
  sm: { width: 80, height: 120 },
  md: { width: 120, height: 180 },
  lg: { width: 180, height: 260 },
  xl: { width: 240, height: 340 },
};

export default function AnimatedCandle({
  size = "md",
  glowing = true,
  className = "",
}: AnimatedCandleProps) {
  const { width, height } = sizes[size];
  const scale = width / 120;

  return (
    <div
      className={`relative inline-flex items-end justify-center ${className}`}
      style={{ width, height }}
    >
      {/* Ambient glow behind candle */}
      {glowing && (
        <div
          className="absolute bottom-0 left-1/2 -translate-x-1/2 rounded-full blur-3xl opacity-30 animate-pulse"
          style={{
            width: width * 1.5,
            height: height * 0.6,
            background: "radial-gradient(ellipse, #ff9d00 0%, #ff6a00 40%, transparent 70%)",
          }}
        />
      )}

      <svg
        viewBox="0 0 120 180"
        width={width}
        height={height}
        xmlns="http://www.w3.org/2000/svg"
        style={{ filter: glowing ? "drop-shadow(0 0 8px rgba(255,157,0,0.6))" : undefined }}
      >
        <defs>
          <radialGradient id="flameGrad" cx="50%" cy="70%" r="50%">
            <stop offset="0%" stopColor="#fff7aa" />
            <stop offset="30%" stopColor="#ffcc00" />
            <stop offset="60%" stopColor="#ff9d00" />
            <stop offset="100%" stopColor="#ff4500" stopOpacity="0" />
          </radialGradient>

          <radialGradient id="innerFlameGrad" cx="50%" cy="80%" r="40%">
            <stop offset="0%" stopColor="#ffffff" />
            <stop offset="50%" stopColor="#fff7aa" />
            <stop offset="100%" stopColor="#ffcc00" stopOpacity="0.5" />
          </radialGradient>

          <linearGradient id="waxGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#c8a96e" />
            <stop offset="20%" stopColor="#f5e6c8" />
            <stop offset="50%" stopColor="#fdf6e3" />
            <stop offset="80%" stopColor="#f5e6c8" />
            <stop offset="100%" stopColor="#a07840" />
          </linearGradient>

          <linearGradient id="holderGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#7a5c2e" />
            <stop offset="20%" stopColor="#cd8b4a" />
            <stop offset="50%" stopColor="#e8a84e" />
            <stop offset="80%" stopColor="#cd8b4a" />
            <stop offset="100%" stopColor="#7a5c2e" />
          </linearGradient>

          <linearGradient id="holderBaseGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#e8a84e" />
            <stop offset="50%" stopColor="#cd8b4a" />
            <stop offset="100%" stopColor="#7a5c2e" />
          </linearGradient>

          <filter id="flameBlur">
            <feGaussianBlur stdDeviation="0.5" />
          </filter>

          <filter id="glowFilter">
            <feGaussianBlur stdDeviation="2" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* === BRASS HOLDER BASE === */}
        {/* Base plate */}
        <ellipse cx="60" cy="175" rx="42" ry="6" fill="url(#holderBaseGrad)" />
        <ellipse cx="60" cy="173" rx="40" ry="4" fill="#e8a84e" opacity="0.6" />

        {/* Stem */}
        <rect x="54" y="155" width="12" height="20" rx="2" fill="url(#holderGrad)" />

        {/* Cup rim */}
        <ellipse cx="60" cy="155" rx="22" ry="5" fill="url(#holderGrad)" />
        <ellipse cx="60" cy="153" rx="20" ry="4" fill="#e8a84e" opacity="0.5" />

        {/* Cup bowl */}
        <path
          d="M38 155 Q38 168 60 170 Q82 168 82 155"
          fill="url(#holderGrad)"
          stroke="#7a5c2e"
          strokeWidth="0.5"
        />

        {/* Holder engraving details */}
        <ellipse cx="60" cy="163" rx="12" ry="2" fill="none" stroke="#e8a84e" strokeWidth="0.5" opacity="0.6" />
        <line x1="45" y1="160" x2="75" y2="160" stroke="#e8a84e" strokeWidth="0.3" opacity="0.4" />

        {/* === CANDLE BODY === */}
        {/* Wax drips */}
        <path d="M48 90 Q46 100 47 120 Q47 130 48 135" stroke="#f0ddb0" strokeWidth="3" fill="none" opacity="0.6" strokeLinecap="round" />
        <path d="M72 85 Q74 95 73 115 Q73 128 72 133" stroke="#f0ddb0" strokeWidth="2.5" fill="none" opacity="0.5" strokeLinecap="round" />

        {/* Main candle body */}
        <rect x="46" y="88" width="28" height="68" rx="3" fill="url(#waxGrad)" />

        {/* Wax top melted pool */}
        <ellipse cx="60" cy="88" rx="14" ry="4" fill="#fdf6e3" />
        <ellipse cx="60" cy="87" rx="12" ry="3" fill="#fff8e7" opacity="0.8" />

        {/* Candle highlight */}
        <rect x="50" y="92" width="5" height="60" rx="2.5" fill="white" opacity="0.12" />

        {/* === WICK === */}
        <line x1="60" y1="88" x2="60" y2="76" stroke="#2d1f0a" strokeWidth="1.5" strokeLinecap="round" />
        <circle cx="60" cy="76" r="1.5" fill="#1a0f00" />

        {/* === FLAME === */}
        <g className="animate-[flicker_1.5s_ease-in-out_infinite_alternate]" style={{ transformOrigin: "60px 68px" }}>
          {/* Outer flame glow */}
          <ellipse cx="60" cy="62" rx="12" ry="18" fill="url(#flameGrad)" opacity="0.4" filter="url(#flameBlur)" />

          {/* Main flame body */}
          <path
            d="M60 44 C55 50 50 56 51 64 C52 71 56 74 60 75 C64 74 68 71 69 64 C70 56 65 50 60 44Z"
            fill="url(#flameGrad)"
          />

          {/* Inner bright flame */}
          <path
            d="M60 54 C57 58 55 62 56 66 C57 70 58.5 72 60 72 C61.5 72 63 70 64 66 C65 62 63 58 60 54Z"
            fill="url(#innerFlameGrad)"
          />

          {/* Flame tip highlight */}
          <ellipse cx="59" cy="56" rx="2" ry="5" fill="white" opacity="0.6" />
        </g>

        {/* Floating embers */}
        <circle cx="54" cy="40" r="1" fill="#ff9d00" opacity="0.8">
          <animate attributeName="cy" values="40;10;10" dur="2s" repeatCount="indefinite" />
          <animate attributeName="opacity" values="0.8;0;0" dur="2s" repeatCount="indefinite" />
          <animate attributeName="cx" values="54;50;50" dur="2s" repeatCount="indefinite" />
        </circle>

        <circle cx="66" cy="42" r="0.8" fill="#ffcc00" opacity="0.7">
          <animate attributeName="cy" values="42;8;8" dur="2.5s" begin="0.5s" repeatCount="indefinite" />
          <animate attributeName="opacity" values="0.7;0;0" dur="2.5s" begin="0.5s" repeatCount="indefinite" />
          <animate attributeName="cx" values="66;70;70" dur="2.5s" begin="0.5s" repeatCount="indefinite" />
        </circle>

        <circle cx="58" cy="38" r="0.6" fill="#ff6a00" opacity="0.6">
          <animate attributeName="cy" values="38;5;5" dur="3s" begin="1s" repeatCount="indefinite" />
          <animate attributeName="opacity" values="0.6;0;0" dur="3s" begin="1s" repeatCount="indefinite" />
          <animate attributeName="cx" values="58;55;55" dur="3s" begin="1s" repeatCount="indefinite" />
        </circle>
      </svg>

      {/* Glow ring on surface */}
      {glowing && (
        <div
          className="absolute bottom-0 left-1/2 -translate-x-1/2 rounded-full blur-xl"
          style={{
            width: width * 0.8,
            height: 12 * scale,
            background: "radial-gradient(ellipse, rgba(255,157,0,0.4) 0%, transparent 70%)",
          }}
        />
      )}
    </div>
  );
}
