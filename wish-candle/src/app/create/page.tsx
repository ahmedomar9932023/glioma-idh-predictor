"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import AnimatedCandle from "@/components/AnimatedCandle";
import Link from "next/link";

const CATEGORIES = [
  { id: "love", label: "Love", icon: "♥" },
  { id: "health", label: "Health", icon: "✚" },
  { id: "career", label: "Career", icon: "⚔" },
  { id: "adventure", label: "Adventure", icon: "✦" },
  { id: "creative", label: "Creative", icon: "✿" },
  { id: "family", label: "Family", icon: "⌂" },
  { id: "wealth", label: "Wealth", icon: "◈" },
  { id: "general", label: "General", icon: "✧" },
];

const REMINDER_OPTIONS = [
  { value: 90, label: "3 months" },
  { value: 180, label: "6 months" },
  { value: 365, label: "1 year" },
  { value: 730, label: "2 years" },
];

export default function CreatePage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [step, setStep] = useState(1);

  const [form, setForm] = useState({
    nickname: "",
    email: "",
    country: "",
    wish_text: "",
    category: "general",
    visibility: "public",
    reminder_days: 365,
  });

  const update = (field: string, value: string | number) => {
    setForm(prev => ({ ...prev, [field]: value }));
  };

  const validate = () => {
    if (!form.nickname.trim()) return "Please enter a name or alias.";
    if (!form.email.trim() || !form.email.includes("@")) return "Please enter a valid email.";
    if (!form.wish_text.trim() || form.wish_text.length < 10) return "Please write a wish (at least 10 characters).";
    return null;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const err = validate();
    if (err) { setError(err); return; }

    setLoading(true);
    setError("");

    try {
      const res = await fetch("/api/create-checkout-session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Something went wrong.");
      if (data.url) window.location.href = data.url;
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to create checkout session.");
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen flex flex-col items-center justify-center px-4 py-16">
      {/* Back link */}
      <div className="w-full max-w-lg mb-6">
        <Link href="/" className="text-amber-700 hover:text-amber-500 text-sm font-cinzel-regular tracking-widest uppercase transition-colors">
          ← Return
        </Link>
      </div>

      <div className="w-full max-w-lg">
        {/* Header */}
        <div className="text-center mb-8">
          <AnimatedCandle size="md" glowing={true} className="mx-auto mb-4" />
          <h1 className="font-cinzel text-3xl font-bold text-gold-shimmer mb-2">
            Speak Your Wish
          </h1>
          <p className="font-fell italic text-amber-300/60 text-sm">
            Your words shall be sealed in flame and remembered.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="stone-panel rounded-xl p-8 space-y-6">
          {/* Nickname */}
          <div>
            <label className="block font-cinzel-regular text-xs tracking-widest text-amber-600 uppercase mb-2">
              Your Name or Alias
            </label>
            <input
              type="text"
              value={form.nickname}
              onChange={e => update("nickname", e.target.value)}
              placeholder="e.g. StarGazer, Anonymous, Luna..."
              maxLength={40}
              className="rpg-input w-full rounded px-4 py-3 text-sm"
            />
          </div>

          {/* Email */}
          <div>
            <label className="block font-cinzel-regular text-xs tracking-widest text-amber-600 uppercase mb-2">
              Your Email
            </label>
            <input
              type="email"
              value={form.email}
              onChange={e => update("email", e.target.value)}
              placeholder="where we send the reminder..."
              className="rpg-input w-full rounded px-4 py-3 text-sm"
            />
            <p className="text-xs text-amber-800/50 mt-1 font-fell italic">
              Used only to deliver your wish — never shared.
            </p>
          </div>

          {/* Country */}
          <div>
            <label className="block font-cinzel-regular text-xs tracking-widest text-amber-600 uppercase mb-2">
              Country <span className="text-amber-800/40">(optional)</span>
            </label>
            <input
              type="text"
              value={form.country}
              onChange={e => update("country", e.target.value)}
              placeholder="e.g. United States, Japan, France..."
              maxLength={60}
              className="rpg-input w-full rounded px-4 py-3 text-sm"
            />
          </div>

          {/* Category */}
          <div>
            <label className="block font-cinzel-regular text-xs tracking-widest text-amber-600 uppercase mb-3">
              Category
            </label>
            <div className="grid grid-cols-4 gap-2">
              {CATEGORIES.map(cat => (
                <button
                  key={cat.id}
                  type="button"
                  onClick={() => update("category", cat.id)}
                  className={`p-2 rounded text-center transition-all duration-200 text-xs font-cinzel-regular border ${
                    form.category === cat.id
                      ? "border-amber-500 bg-amber-900/30 text-amber-300"
                      : "border-amber-900/30 bg-black/20 text-amber-700 hover:border-amber-700"
                  }`}
                >
                  <div className="text-base mb-1">{cat.icon}</div>
                  {cat.label}
                </button>
              ))}
            </div>
          </div>

          {/* Wish text */}
          <div>
            <label className="block font-cinzel-regular text-xs tracking-widest text-amber-600 uppercase mb-2">
              Your Wish
            </label>
            <textarea
              value={form.wish_text}
              onChange={e => update("wish_text", e.target.value)}
              placeholder="Speak your deepest desire into the flame..."
              rows={5}
              maxLength={1000}
              className="rpg-input w-full rounded px-4 py-3 text-sm font-fell italic leading-relaxed resize-none"
            />
            <p className="text-right text-xs text-amber-800/40 mt-1">
              {form.wish_text.length}/1000
            </p>
          </div>

          {/* Reminder date */}
          <div>
            <label className="block font-cinzel-regular text-xs tracking-widest text-amber-600 uppercase mb-3">
              Remind Me In
            </label>
            <div className="grid grid-cols-4 gap-2">
              {REMINDER_OPTIONS.map(opt => (
                <button
                  key={opt.value}
                  type="button"
                  onClick={() => update("reminder_days", opt.value)}
                  className={`p-2 rounded text-center transition-all duration-200 text-xs font-cinzel-regular border ${
                    form.reminder_days === opt.value
                      ? "border-amber-500 bg-amber-900/30 text-amber-300"
                      : "border-amber-900/30 bg-black/20 text-amber-700 hover:border-amber-700"
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {/* Visibility */}
          <div>
            <label className="block font-cinzel-regular text-xs tracking-widest text-amber-600 uppercase mb-3">
              Visibility
            </label>
            <div className="flex gap-3">
              <button
                type="button"
                onClick={() => update("visibility", "public")}
                className={`flex-1 p-3 rounded text-xs font-cinzel-regular border transition-all ${
                  form.visibility === "public"
                    ? "border-amber-500 bg-amber-900/30 text-amber-300"
                    : "border-amber-900/30 bg-black/20 text-amber-700 hover:border-amber-700"
                }`}
              >
                ✦ Public
                <p className="text-amber-800/50 font-fell normal-case text-xs mt-1">Shown on the world wall</p>
              </button>
              <button
                type="button"
                onClick={() => update("visibility", "private")}
                className={`flex-1 p-3 rounded text-xs font-cinzel-regular border transition-all ${
                  form.visibility === "private"
                    ? "border-amber-500 bg-amber-900/30 text-amber-300"
                    : "border-amber-900/30 bg-black/20 text-amber-700 hover:border-amber-700"
                }`}
              >
                ◈ Private
                <p className="text-amber-800/50 font-fell normal-case text-xs mt-1">Only you can see it</p>
              </button>
            </div>
          </div>

          {error && (
            <div className="bg-red-950/40 border border-red-800/40 rounded p-3">
              <p className="text-red-400 text-sm font-fell">{error}</p>
            </div>
          )}

          {/* Submit */}
          <button
            type="submit"
            disabled={loading}
            className="rpg-btn w-full py-4 rounded text-base disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Opening the Portal…" : "Light the Candle — $0.99"}
          </button>

          <p className="text-center text-xs text-amber-800/40 font-fell italic">
            Secured by Stripe. No card data touches our servers.
          </p>
        </form>
      </div>
    </main>
  );
}
