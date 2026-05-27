import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function GET(req: NextRequest) {
  const authHeader = req.headers.get("authorization");
  const cronSecret = process.env.CRON_SECRET;

  if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const tomorrow = new Date(today);
  tomorrow.setDate(tomorrow.getDate() + 1);

  const wishes = await prisma.wish.findMany({
    where: {
      payment_status: "paid",
      status: "burning",
      reminder_sent: false,
      reminder_date: {
        gte: today,
        lt: tomorrow,
      },
    },
  });

  const results: { id: number; public_id: string; sent: boolean; error?: string }[] = [];

  for (const wish of wishes) {
    try {
      await sendReminderEmail(wish);
      await prisma.wish.update({
        where: { id: wish.id },
        data: { reminder_sent: true },
      });
      results.push({ id: wish.id, public_id: wish.public_id, sent: true });
    } catch (err) {
      console.error(`Failed to send reminder for wish ${wish.id}:`, err);
      results.push({
        id: wish.id,
        public_id: wish.public_id,
        sent: false,
        error: err instanceof Error ? err.message : "Unknown",
      });
    }
  }

  return NextResponse.json({
    processed: wishes.length,
    results,
  });
}

async function sendReminderEmail(wish: {
  email: string;
  nickname: string;
  wish_text: string;
  public_id: string;
}) {
  const siteUrl = process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000";
  const wishUrl = `${siteUrl}/wish/${wish.public_id}`;
  const respondUrl = `${siteUrl}/wish/${wish.public_id}/respond`;

  const resendKey = process.env.RESEND_API_KEY;
  if (!resendKey) {
    console.log(`[DEV] Would send reminder to ${wish.email} for wish ${wish.public_id}`);
    return;
  }

  const { Resend } = await import("resend");
  const resend = new Resend(resendKey);

  await resend.emails.send({
    from: "Wish Candle <wishes@wishcandle.com>",
    to: wish.email,
    subject: "Your candle is still burning — did your wish come true?",
    html: `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body { background: #0a0905; color: #e8d5a3; font-family: Georgia, serif; margin: 0; padding: 0; }
    .container { max-width: 560px; margin: 0 auto; padding: 40px 24px; }
    .title { font-size: 28px; color: #fbbf24; margin-bottom: 8px; }
    .divider { border: none; border-top: 1px solid #3a2810; margin: 24px 0; }
    .wish-text { font-style: italic; font-size: 18px; color: #fef3c7; padding: 20px; background: #1a1510; border-left: 3px solid #d97706; margin: 24px 0; }
    .btn { display: inline-block; padding: 14px 28px; background: linear-gradient(135deg, #d97706, #92400e); color: #fffbeb; text-decoration: none; border-radius: 4px; font-weight: bold; letter-spacing: 0.05em; margin: 8px 4px; }
    .footer { color: #78350f; font-size: 12px; margin-top: 40px; }
  </style>
</head>
<body>
  <div class="container">
    <p style="color:#d97706; font-size:12px; letter-spacing:0.3em; text-transform:uppercase;">✦ The Candle Remembers ✦</p>
    <h1 class="title">Dear ${wish.nickname},</h1>
    <p>Time has passed since you whispered a wish into the flame.</p>
    <p>The candle has been burning faithfully, holding your words:</p>

    <div class="wish-text">"${wish.wish_text}"</div>

    <p>Now the candle asks — <strong style="color:#fbbf24;">did your wish come true?</strong></p>
    <hr class="divider">

    <a href="${respondUrl}?status=fulfilled" class="btn">✓ Yes, it came true!</a>
    <a href="${respondUrl}?status=not_yet" class="btn" style="background: linear-gradient(135deg, #374151, #1f2937);">◈ Not yet...</a>

    <hr class="divider">
    <p style="color:#92400e; font-style:italic; font-size:14px;">
      "Every wish spoken to the flame becomes a spark that never truly dies."
    </p>

    <div class="footer">
      <p>View your candle: <a href="${wishUrl}" style="color:#d97706;">${wishUrl}</a></p>
      <p>This email was sent because you lit a candle at Wish Candle.</p>
    </div>
  </div>
</body>
</html>
    `,
  });
}
