import { notFound } from "next/navigation";
import Link from "next/link";
import { prisma } from "@/lib/prisma";
import AnimatedCandle from "@/components/AnimatedCandle";

const CATEGORY_ICONS: Record<string, string> = {
  love: "♥", health: "✚", career: "⚔", adventure: "✦",
  creative: "✿", family: "⌂", wealth: "◈", general: "✧",
};

const STATUS_INFO: Record<string, { label: string; color: string; desc: string }> = {
  burning: { label: "Still Burning", color: "text-amber-400", desc: "This wish burns faithfully in the registry." },
  fulfilled: { label: "Fulfilled ✓", color: "text-emerald-400", desc: "This wish came true." },
  not_yet: { label: "Not Yet", color: "text-slate-400", desc: "The journey continues." },
};

export default async function WishPage({ params }: { params: Promise<{ public_id: string }> }) {
  const { public_id } = await params;

  const wish = await prisma.wish.findUnique({
    where: { public_id },
  });

  if (!wish || wish.payment_status !== "paid") notFound();

  const icon = CATEGORY_ICONS[wish.category] ?? "✧";
  const statusInfo = STATUS_INFO[wish.status] ?? STATUS_INFO.burning;
  const createdDate = new Date(wish.created_at).toLocaleDateString("en-US", {
    year: "numeric", month: "long", day: "numeric",
  });
  const reminderDate = new Date(wish.reminder_date).toLocaleDateString("en-US", {
    year: "numeric", month: "long", day: "numeric",
  });

  return (
    <main className="min-h-screen flex flex-col items-center justify-center px-4 py-16">
      {/* Back */}
      <div className="w-full max-w-lg mb-6">
        <Link href="/world-wishes" className="text-amber-700 hover:text-amber-500 text-sm font-cinzel-regular tracking-widest uppercase transition-colors">
          ← All Wishes
        </Link>
      </div>

      <div className="w-full max-w-lg">
        {/* Candle */}
        <div className="text-center mb-6">
          <AnimatedCandle size="lg" glowing={true} className="mx-auto mb-3" />
          <p className="font-cinzel-regular text-xs tracking-[0.4em] text-amber-600/60 uppercase">
            {icon} {wish.category}
          </p>
        </div>

        {/* Card */}
        <div className="stone-panel rounded-xl p-8">
          {/* Status */}
          <div className="flex items-center justify-between mb-6">
            <span className={`font-cinzel-regular text-sm ${statusInfo.color}`}>
              {statusInfo.label}
            </span>
            <span className="text-amber-800/40 text-xs font-fell">{statusInfo.desc}</span>
          </div>

          <div className="ornament-divider mb-6">
            <span className="text-amber-800/30 text-xs">✦</span>
          </div>

          {/* Wish text */}
          <blockquote className="font-fell italic text-xl text-amber-100/90 leading-relaxed text-center mb-8">
            &ldquo;{wish.wish_text}&rdquo;
          </blockquote>

          <div className="ornament-divider mb-6">
            <span className="text-amber-800/30 text-xs">✦</span>
          </div>

          {/* Meta */}
          <div className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="font-cinzel-regular text-amber-700/60 text-xs uppercase tracking-wider">Wisher</span>
              <span className="font-fell text-amber-300">{wish.nickname}</span>
            </div>
            {wish.country && (
              <div className="flex justify-between">
                <span className="font-cinzel-regular text-amber-700/60 text-xs uppercase tracking-wider">Country</span>
                <span className="font-fell text-amber-300">{wish.country}</span>
              </div>
            )}
            <div className="flex justify-between">
              <span className="font-cinzel-regular text-amber-700/60 text-xs uppercase tracking-wider">Lit on</span>
              <span className="font-fell text-amber-300">{createdDate}</span>
            </div>
            <div className="flex justify-between">
              <span className="font-cinzel-regular text-amber-700/60 text-xs uppercase tracking-wider">Reminder</span>
              <span className="font-fell text-amber-300">{reminderDate}</span>
            </div>
          </div>

          {/* Respond link */}
          {wish.status === "burning" && (
            <div className="mt-8 pt-6 border-t border-amber-900/30 text-center">
              <p className="font-fell italic text-amber-800/50 text-xs mb-3">Is this your candle?</p>
              <Link href={`/wish/${wish.public_id}/respond`}>
                <button className="text-amber-600 hover:text-amber-400 font-cinzel-regular text-xs tracking-widest uppercase transition-colors">
                  Mark as Fulfilled or Not Yet →
                </button>
              </Link>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
