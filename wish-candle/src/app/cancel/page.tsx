import Link from "next/link";
import AnimatedCandle from "@/components/AnimatedCandle";

export default function CancelPage() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center px-4 text-center">
      <div className="max-w-md mx-auto">
        {/* Dimmed candle */}
        <div className="opacity-40 mb-6">
          <AnimatedCandle size="lg" glowing={false} className="mx-auto" />
        </div>

        <p className="font-cinzel-regular text-amber-700/60 text-xs tracking-[0.5em] uppercase mb-4">
          ✦ The Flame Flickers ✦
        </p>

        <h1 className="font-cinzel text-3xl font-bold text-amber-700 mb-4">
          Wish Not Yet Spoken
        </h1>

        <div className="ornament-divider w-32 mx-auto mb-6">
          <span className="text-amber-800/40 text-xs">✦</span>
        </div>

        <p className="font-fell italic text-amber-400/60 text-lg mb-3">
          Your payment was cancelled.
          The candle remains unlit, waiting for your return.
        </p>

        <p className="font-fell text-amber-800/40 text-sm mb-10">
          No charge was made. Your wish still awaits.
        </p>

        <div className="flex flex-col gap-3">
          <Link href="/create">
            <button className="rpg-btn w-full py-3 rounded">
              Try Again
            </button>
          </Link>
          <Link href="/">
            <button className="text-amber-800/50 hover:text-amber-700 font-cinzel-regular text-xs tracking-widest uppercase transition-colors">
              Return Home
            </button>
          </Link>
        </div>
      </div>
    </main>
  );
}
