"use client";

import { useState, use } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import AnimatedCandle from "@/components/AnimatedCandle";

export default function RespondPage({ params }: { params: Promise<{ public_id: string }> }) {
  const { public_id } = use(params);
  const router = useRouter();
  const [choice, setChoice] = useState<"fulfilled" | "not_yet" | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async () => {
    if (!choice) return;
    setLoading(true);
    setError("");

    try {
      const res = await fetch(`/api/wishes`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ public_id, status: choice }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || "Failed to update.");
      }

      router.push(`/wish/${public_id}`);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen flex flex-col items-center justify-center px-4 py-16">
      <div className="w-full max-w-md mb-6">
        <Link href={`/wish/${public_id}`} className="text-amber-700 hover:text-amber-500 text-sm font-cinzel-regular tracking-widest uppercase transition-colors">
          ← Back to Wish
        </Link>
      </div>

      <div className="w-full max-w-md text-center">
        <AnimatedCandle size="lg" glowing className="mx-auto mb-6" />

        <p className="font-cinzel-regular text-amber-600/60 text-xs tracking-[0.4em] uppercase mb-4">
          ✦ The Candle Asks ✦
        </p>

        <h1 className="font-cinzel text-3xl font-bold text-gold-shimmer mb-4">
          Did Your Wish Come True?
        </h1>

        <p className="font-fell italic text-amber-300/60 text-base mb-10">
          The flame has waited patiently. Now it seeks an answer.
        </p>

        <div className="space-y-4 mb-8">
          <button
            onClick={() => setChoice("fulfilled")}
            className={`w-full p-6 rounded-lg border-2 transition-all duration-200 text-left ${
              choice === "fulfilled"
                ? "border-emerald-500 bg-emerald-950/30"
                : "border-amber-900/30 bg-black/20 hover:border-emerald-700"
            }`}
          >
            <div className="flex items-center gap-4">
              <span className="text-3xl">✓</span>
              <div>
                <p className="font-cinzel-regular text-emerald-400 text-sm tracking-wider">Yes, it came true!</p>
                <p className="font-fell italic text-emerald-600/60 text-sm mt-1">
                  The wish was fulfilled. The candle may rest in peace.
                </p>
              </div>
            </div>
          </button>

          <button
            onClick={() => setChoice("not_yet")}
            className={`w-full p-6 rounded-lg border-2 transition-all duration-200 text-left ${
              choice === "not_yet"
                ? "border-amber-500 bg-amber-950/30"
                : "border-amber-900/30 bg-black/20 hover:border-amber-700"
            }`}
          >
            <div className="flex items-center gap-4">
              <span className="text-3xl text-amber-600">◈</span>
              <div>
                <p className="font-cinzel-regular text-amber-400 text-sm tracking-wider">Not yet...</p>
                <p className="font-fell italic text-amber-600/60 text-sm mt-1">
                  The journey continues. Perhaps next year.
                </p>
              </div>
            </div>
          </button>
        </div>

        {error && (
          <p className="text-red-400 text-sm font-fell mb-4">{error}</p>
        )}

        <button
          onClick={handleSubmit}
          disabled={!choice || loading}
          className="rpg-btn w-full py-4 rounded disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {loading ? "Updating the Registry…" : "Seal the Answer"}
        </button>
      </div>
    </main>
  );
}
