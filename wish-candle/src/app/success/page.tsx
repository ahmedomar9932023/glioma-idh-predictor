"use client";

import { useEffect, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import AnimatedCandle from "@/components/AnimatedCandle";
import CelebrationEffects from "@/components/CelebrationEffects";

function SuccessContent() {
  const params = useSearchParams();
  const sessionId = params.get("session_id");
  const [publicId, setPublicId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [showContent, setShowContent] = useState(false);

  useEffect(() => {
    if (!sessionId) { setLoading(false); return; }

    const poll = async (attempts = 0) => {
      try {
        const res = await fetch(`/api/wishes?session_id=${sessionId}`);
        if (res.ok) {
          const data = await res.json();
          if (data.public_id) {
            setPublicId(data.public_id);
            setLoading(false);
            setTimeout(() => setShowContent(true), 300);
            return;
          }
        }
      } catch { /* ignore */ }

      if (attempts < 8) {
        setTimeout(() => poll(attempts + 1), 1500);
      } else {
        setLoading(false);
        setTimeout(() => setShowContent(true), 300);
      }
    };

    poll();
  }, [sessionId]);

  return (
    <main className="min-h-screen flex flex-col items-center justify-center px-4 relative overflow-hidden">
      <CelebrationEffects />

      {/* Large ambient glow */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div
          className="w-[800px] h-[800px] rounded-full animate-pulse"
          style={{
            background: "radial-gradient(circle, rgba(255,157,0,0.08) 0%, transparent 70%)",
          }}
        />
      </div>

      <div
        className={`relative z-10 text-center max-w-lg mx-auto transition-all duration-1000 ${
          showContent ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"
        }`}
      >
        {/* Ornament */}
        <p className="font-cinzel-regular text-amber-600/60 text-xs tracking-[0.5em] uppercase mb-6">
          ✦ The Flame Accepts ✦
        </p>

        {/* Big candle */}
        <div className="mb-6 relative">
          <AnimatedCandle size="xl" glowing={true} className="mx-auto" />

          {/* Floating hearts around candle */}
          {[...Array(6)].map((_, i) => (
            <div
              key={i}
              className="absolute text-pink-400/60 text-xl pointer-events-none"
              style={{
                left: `${20 + i * 12}%`,
                top: `${10 + (i % 3) * 25}%`,
                animation: `heartRise ${2 + i * 0.3}s ease-out ${i * 0.4}s forwards`,
              }}
            >
              ♥
            </div>
          ))}
        </div>

        {/* Title */}
        <h1 className="font-cinzel text-4xl md:text-5xl font-bold text-gold-shimmer mb-4">
          Your Wish is Sealed
        </h1>

        <div className="ornament-divider w-48 mx-auto mb-6">
          <span className="text-amber-600/40 text-xs">✦</span>
        </div>

        <p className="font-fell italic text-xl text-amber-200/80 mb-3 leading-relaxed">
          The flame has accepted your offering.
          Your wish burns now in the ancient registry.
        </p>

        <p className="font-fell text-amber-500/60 text-base mb-8">
          A confirmation has been sent to your email.
          When the time comes, the candle shall ask — did it come true?
        </p>

        {/* Spark divider */}
        <div className="flex justify-center gap-3 mb-8 text-amber-600/40 text-sm">
          <span>✦</span><span>✦</span><span>✦</span>
        </div>

        {/* Buttons */}
        <div className="flex flex-col gap-3">
          {publicId && (
            <Link href={`/wish/${publicId}`}>
              <button className="rpg-btn w-full py-3 rounded">
                View Your Candle
              </button>
            </Link>
          )}
          <Link href="/world-wishes">
            <button className="w-full py-3 rounded border border-amber-900/40 text-amber-600 hover:text-amber-400 hover:border-amber-600 font-cinzel-regular text-sm tracking-widest uppercase transition-all duration-200">
              See Other Wishes
            </button>
          </Link>
          <Link href="/">
            <button className="text-amber-800/50 hover:text-amber-700 font-cinzel-regular text-xs tracking-widest uppercase transition-colors">
              Return Home
            </button>
          </Link>
        </div>

        {/* Lore */}
        <div className="mt-12">
          <div className="ornament-divider mb-4">
            <span className="text-amber-900/30 text-xs">✦</span>
          </div>
          <p className="font-fell italic text-amber-900/40 text-xs">
            &ldquo;Every wish spoken to the flame becomes a spark that never truly dies.&rdquo;
          </p>
        </div>
      </div>

      {/* Loading overlay */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-stone-950/80 z-20">
          <div className="text-center">
            <AnimatedCandle size="lg" glowing={true} className="mx-auto mb-4" />
            <p className="font-cinzel-regular text-amber-500 text-sm tracking-widest animate-pulse">
              Sealing your wish…
            </p>
          </div>
        </div>
      )}
    </main>
  );
}

export default function SuccessPage() {
  return (
    <Suspense fallback={
      <main className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <AnimatedCandle size="lg" glowing className="mx-auto mb-4" />
          <p className="font-cinzel-regular text-amber-500 text-sm tracking-widest animate-pulse">Loading…</p>
        </div>
      </main>
    }>
      <SuccessContent />
    </Suspense>
  );
}
