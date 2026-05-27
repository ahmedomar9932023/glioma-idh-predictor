import Link from "next/link";
import { prisma } from "@/lib/prisma";
import WishCard from "@/components/WishCard";

const CATEGORIES = [
  { id: "all", label: "All", icon: "✧" },
  { id: "love", label: "Love", icon: "♥" },
  { id: "health", label: "Health", icon: "✚" },
  { id: "career", label: "Career", icon: "⚔" },
  { id: "adventure", label: "Adventure", icon: "✦" },
  { id: "creative", label: "Creative", icon: "✿" },
  { id: "family", label: "Family", icon: "⌂" },
  { id: "wealth", label: "Wealth", icon: "◈" },
  { id: "general", label: "General", icon: "✧" },
];

export default async function WorldWishesPage({
  searchParams,
}: {
  searchParams: Promise<{ category?: string; page?: string }>;
}) {
  const params = await searchParams;
  const category = params.category ?? "all";
  const page = Math.max(1, parseInt(params.page ?? "1", 10));
  const perPage = 12;

  const where = {
    payment_status: "paid",
    visibility: "public",
    ...(category !== "all" ? { category } : {}),
  };

  const [wishes, total] = await Promise.all([
    prisma.wish.findMany({
      where,
      orderBy: { created_at: "desc" },
      skip: (page - 1) * perPage,
      take: perPage,
    }),
    prisma.wish.count({ where }),
  ]);

  const totalPages = Math.ceil(total / perPage);

  return (
    <main className="min-h-screen px-4 py-16">
      {/* Header */}
      <div className="max-w-5xl mx-auto text-center mb-12">
        <Link href="/" className="text-amber-700 hover:text-amber-500 text-sm font-cinzel-regular tracking-widest uppercase transition-colors block mb-8">
          ← Return to the Shrine
        </Link>

        <p className="font-cinzel-regular text-amber-600/60 text-xs tracking-[0.4em] uppercase mb-4">
          ✦ The Great Registry ✦
        </p>
        <h1 className="font-cinzel text-4xl md:text-5xl font-bold text-gold-shimmer mb-4">
          World&apos;s Wishes
        </h1>
        <p className="font-fell italic text-amber-300/60 text-lg mb-2">
          Every flame tells a story. Every wish, a hope.
        </p>
        <p className="font-cinzel-regular text-amber-800 text-sm">
          {total.toLocaleString()} candles burning
        </p>
      </div>

      {/* Category filter */}
      <div className="max-w-5xl mx-auto mb-8">
        <div className="flex flex-wrap gap-2 justify-center">
          {CATEGORIES.map(cat => (
            <Link
              key={cat.id}
              href={`/world-wishes?category=${cat.id}`}
            >
              <button
                className={`px-4 py-2 rounded text-xs font-cinzel-regular border transition-all ${
                  category === cat.id
                    ? "border-amber-500 bg-amber-900/30 text-amber-300"
                    : "border-amber-900/30 bg-black/20 text-amber-700 hover:border-amber-700"
                }`}
              >
                {cat.icon} {cat.label}
              </button>
            </Link>
          ))}
        </div>
      </div>

      {/* Grid */}
      <div className="max-w-5xl mx-auto">
        {wishes.length === 0 ? (
          <div className="text-center py-20">
            <p className="font-fell italic text-amber-800/50 text-xl">
              No wishes in this category yet. Be the first.
            </p>
            <Link href="/create" className="mt-6 inline-block">
              <button className="rpg-btn px-8 py-3 rounded text-sm mt-4">
                Light a Candle
              </button>
            </Link>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {wishes.map(wish => (
              <WishCard key={wish.id} wish={wish} />
            ))}
          </div>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="max-w-5xl mx-auto mt-10 flex justify-center gap-3">
          {page > 1 && (
            <Link href={`/world-wishes?category=${category}&page=${page - 1}`}>
              <button className="px-4 py-2 rounded border border-amber-900/40 text-amber-600 hover:border-amber-600 font-cinzel-regular text-xs tracking-widest uppercase transition-all">
                ← Prev
              </button>
            </Link>
          )}
          <span className="px-4 py-2 font-cinzel-regular text-amber-700 text-xs">
            {page} / {totalPages}
          </span>
          {page < totalPages && (
            <Link href={`/world-wishes?category=${category}&page=${page + 1}`}>
              <button className="px-4 py-2 rounded border border-amber-900/40 text-amber-600 hover:border-amber-600 font-cinzel-regular text-xs tracking-widest uppercase transition-all">
                Next →
              </button>
            </Link>
          )}
        </div>
      )}

      {/* CTA */}
      <div className="max-w-5xl mx-auto text-center mt-16">
        <div className="ornament-divider mb-6">
          <span className="text-amber-800/30 text-xs">✦</span>
        </div>
        <p className="font-fell italic text-amber-800/40 text-sm mb-4">
          Add your own wish to the registry
        </p>
        <Link href="/create">
          <button className="rpg-btn px-8 py-3 rounded text-sm">
            Light Your Candle — $0.99
          </button>
        </Link>
      </div>
    </main>
  );
}
