import Link from "next/link";
import AnimatedCandle from "./AnimatedCandle";

interface WishCardProps {
  wish: {
    public_id: string;
    nickname: string;
    country: string;
    wish_text: string;
    category: string;
    created_at: Date | string;
    status: string;
  };
}

const CATEGORY_ICONS: Record<string, string> = {
  love: "♥",
  health: "✚",
  career: "⚔",
  adventure: "✦",
  creative: "✿",
  family: "⌂",
  wealth: "◈",
  general: "✧",
};

const STATUS_COLORS: Record<string, string> = {
  burning: "text-amber-400",
  fulfilled: "text-emerald-400",
  not_yet: "text-slate-400",
};

const STATUS_LABELS: Record<string, string> = {
  burning: "Still Burning",
  fulfilled: "Fulfilled",
  not_yet: "Not Yet",
};

export default function WishCard({ wish }: WishCardProps) {
  const icon = CATEGORY_ICONS[wish.category] ?? "✧";
  const date = new Date(wish.created_at);
  const formattedDate = date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  const preview = wish.wish_text.length > 90 ? wish.wish_text.slice(0, 87) + "…" : wish.wish_text;

  return (
    <Link href={`/wish/${wish.public_id}`}>
      <div className="wish-card stone-panel rounded-lg p-4 cursor-pointer group relative overflow-hidden">
        {/* Category badge */}
        <div className="flex items-center justify-between mb-3">
          <span className="text-gold-400 text-lg font-cinzel-regular">{icon}</span>
          <span className={`text-xs font-cinzel-regular ${STATUS_COLORS[wish.status] ?? "text-amber-400"}`}>
            {STATUS_LABELS[wish.status] ?? wish.status}
          </span>
        </div>

        {/* Wish text */}
        <p className="font-fell italic text-amber-100/90 text-sm leading-relaxed mb-3 min-h-[60px]">
          &ldquo;{preview}&rdquo;
        </p>

        {/* Footer */}
        <div className="flex items-center justify-between pt-2 border-t border-amber-900/30">
          <div className="flex items-center gap-2">
            <AnimatedCandle size="sm" glowing={false} />
            <div>
              <p className="text-xs font-cinzel-regular text-gold-400">{wish.nickname}</p>
              {wish.country && (
                <p className="text-xs text-amber-800">{wish.country}</p>
              )}
            </div>
          </div>
          <p className="text-xs text-amber-800/60">{formattedDate}</p>
        </div>

        {/* Hover glow */}
        <div className="absolute inset-0 bg-gradient-to-t from-amber-900/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none rounded-lg" />
      </div>
    </Link>
  );
}
