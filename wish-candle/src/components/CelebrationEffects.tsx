"use client";

import { useEffect, useState, useCallback } from "react";

interface Particle {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  type: "heart" | "spark" | "star" | "ember";
  color: string;
  size: number;
  life: number;
  maxLife: number;
  rotation: number;
  rotationSpeed: number;
}

const COLORS = {
  gold: ["#fbbf24", "#fcd34d", "#f59e0b", "#d97706"],
  fire: ["#ff9d00", "#ff6a00", "#ff4500", "#ffcc00"],
  magic: ["#e879f9", "#a855f7", "#ec4899", "#06b6d4"],
};

function Heart({ x, y, size, opacity, color }: { x: number; y: number; size: number; opacity: number; color: string }) {
  return (
    <svg
      style={{ position: "absolute", left: x - size, top: y - size, width: size * 2, height: size * 2, opacity, pointerEvents: "none" }}
      viewBox="0 0 24 24"
    >
      <path
        d="M12 21.593c-5.63-5.539-11-10.297-11-14.402 0-3.791 3.068-5.191 5.281-5.191 1.312 0 4.151.501 5.719 4.457 1.59-3.968 4.464-4.447 5.726-4.447 2.54 0 5.274 1.621 5.274 5.181 0 4.069-5.136 8.625-11 14.402z"
        fill={color}
      />
    </svg>
  );
}

function Star({ x, y, size, opacity, color, rotation }: { x: number; y: number; size: number; opacity: number; color: string; rotation: number }) {
  return (
    <svg
      style={{
        position: "absolute", left: x - size, top: y - size, width: size * 2, height: size * 2,
        opacity, transform: `rotate(${rotation}deg)`, pointerEvents: "none"
      }}
      viewBox="0 0 24 24"
    >
      <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" fill={color} />
    </svg>
  );
}

export default function CelebrationEffects() {
  const [particles, setParticles] = useState<Particle[]>([]);
  const [frame, setFrame] = useState(0);

  const spawnParticle = useCallback((id: number): Particle => {
    const cx = window.innerWidth / 2;
    const cy = window.innerHeight / 2;
    const angle = Math.random() * Math.PI * 2;
    const speed = 2 + Math.random() * 4;
    const types: Particle["type"][] = ["heart", "spark", "star", "ember"];
    const type = types[Math.floor(Math.random() * types.length)];
    const colorArr = type === "heart" ? COLORS.magic : type === "star" ? COLORS.gold : COLORS.fire;
    const color = colorArr[Math.floor(Math.random() * colorArr.length)];

    return {
      id,
      x: cx + (Math.random() - 0.5) * 100,
      y: cy + (Math.random() - 0.5) * 100,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed - 3,
      type,
      color,
      size: 8 + Math.random() * 14,
      life: 0,
      maxLife: 80 + Math.random() * 60,
      rotation: Math.random() * 360,
      rotationSpeed: (Math.random() - 0.5) * 6,
    };
  }, []);

  useEffect(() => {
    let counter = 0;
    const interval = setInterval(() => {
      setParticles(prev => {
        const updated = prev
          .map(p => ({
            ...p,
            x: p.x + p.vx,
            y: p.y + p.vy,
            vy: p.vy + 0.08,
            vx: p.vx * 0.99,
            life: p.life + 1,
            rotation: p.rotation + p.rotationSpeed,
          }))
          .filter(p => p.life < p.maxLife);

        const newParticles: Particle[] = [];
        const spawnCount = prev.length < 60 ? 3 : 1;
        for (let i = 0; i < spawnCount; i++) {
          newParticles.push(spawnParticle(counter++));
        }

        return [...updated, ...newParticles];
      });
    }, 50);

    const timeout = setTimeout(() => clearInterval(interval), 6000);

    return () => {
      clearInterval(interval);
      clearTimeout(timeout);
    };
  }, [spawnParticle]);

  return (
    <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 50, overflow: "hidden" }}>
      {particles.map(p => {
        const opacity = Math.max(0, 1 - p.life / p.maxLife);

        if (p.type === "heart") {
          return <Heart key={p.id} x={p.x} y={p.y} size={p.size} opacity={opacity} color={p.color} />;
        }
        if (p.type === "star") {
          return <Star key={p.id} x={p.x} y={p.y} size={p.size} opacity={opacity} color={p.color} rotation={p.rotation} />;
        }

        return (
          <div
            key={p.id}
            style={{
              position: "absolute",
              left: p.x,
              top: p.y,
              width: p.type === "ember" ? p.size * 0.4 : p.size * 0.5,
              height: p.type === "ember" ? p.size * 0.4 : p.size * 0.5,
              borderRadius: "50%",
              background: p.color,
              opacity,
              boxShadow: `0 0 ${p.size}px ${p.color}`,
              transform: `rotate(${p.rotation}deg)`,
              pointerEvents: "none",
            }}
          />
        );
      })}
    </div>
  );
}
