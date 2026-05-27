import Link from "next/link";
import AnimatedCandle from "@/components/AnimatedCandle";
import { prisma } from "@/lib/prisma";

async function getStats() {
  try {
    const total = await prisma.wish.count({ where: { payment_status: "paid" } });
    const fulfilled = await prisma.wish.count({ where: { payment_status: "paid", status: "fulfilled" } });
    return { total, fulfilled };
  } catch {
    return { total: 0, fulfilled: 0 };
  }
}

export default async function HomePage() {
  const { total, fulfilled } = await getStats();

  return (
    <main className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden">
      {/* Ambient light behind scene */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="w-[600px] h-[600px] rounded-full opacity-10"
          style={{ background: "radial-gradient(circle, #ff9d00 0%, transparent 70%)" }} />
      </div>

      {/* Floating dust particles */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        {[...Array(12)].map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 rounded-full bg-amber-400/20"
            style={{
              left: `${10 + i * 7.5}%`,
              top: `${20 + (i % 5) * 15}%`,
              animation: `drift ${3 + (i % 4)}s ease-in-out ${i * 0.5}s infinite alternate`,
            }}
          />
        ))}
      </div>

      <div className="relative z-10 flex flex-col items-center text-center px-4 max-w-3xl mx-auto">
        {/* Ornament top */}
        <p className="font-cinzel-regular text-amber-600/60 text-sm tracking-[0.4em] uppercase mb-6">
          ✦ Ancient Ritual ✦
        </p>

        {/* Main candle */}
        <div className="mb-6">
          <AnimatedCandle size="xl" glowing={true} />
        </div>

        {/* Title */}
        <h1 className="font-cinzel text-5xl md:text-6xl font-bold text-gold-shimmer mb-4 leading-tight">
          Wish Candle
        </h1>

        <div className="ornament-divider w-48 mx-auto mb-6">
          <span className="text-amber-600/60 text-xs">✦</span>
        </div>

        {/* Tagline */}
        <p className="font-fell italic text-xl md:text-2xl text-amber-200/80 mb-2">
          Light a candle. Speak your wish into the flame.
        </p>
        <p className="font-fell text-amber-400/60 text-base mb-10">
          One year from now, the candle asks — did it come true?
        </p>

        {/* Stats */}
        {total > 0 && (
          <div className="flex gap-8 mb-10">
            <div className="text-center">
              <p className="font-cinzel text-2xl font-bold text-gold-400">{total.toLocaleString()}</p>
              <p className="text-xs text-amber-700 font-cinzel-regular tracking-wider uppercase">Candles Lit</p>
            </div>
            <div className="w-px bg-amber-900/40" />
            <div className="text-center">
              <p className="font-cinzel text-2xl font-bold text-emerald-400">{fulfilled.toLocaleString()}</p>
              <p className="text-xs text-amber-700 font-cinzel-regular tracking-wider uppercase">Wishes Fulfilled</p>
            </div>
          </div>
        )}

        {/* CTA */}
        <Link href="/create">
          <button className="rpg-btn px-10 py-4 rounded text-base mb-4">
            Light Your Candle — $0.99
          </button>
        </Link>

        <Link href="/world-wishes">
          <button className="text-amber-600 hover:text-amber-400 font-cinzel-regular text-sm tracking-widest uppercase transition-colors duration-200">
            View the World&apos;s Wishes →
          </button>
        </Link>

        {/* Lore text */}
        <div className="mt-16 max-w-md mx-auto">
          <div className="ornament-divider mb-4">
            <span className="text-amber-800/40 text-xs">✦</span>
          </div>
          <p className="font-fell italic text-amber-800/50 text-sm leading-relaxed">
            &ldquo;Since time immemorial, the faithful have spoken their deepest desires into flame.
            The candle remembers what the heart dares to hope.&rdquo;
          </p>
        </div>
      </div>

      {/* Bottom fade */}
      <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-stone-950 to-transparent pointer-events-none" />
    </main>
  );
}
