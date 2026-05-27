import { NextRequest, NextResponse } from "next/server";
import { stripe } from "@/lib/stripe";
import { prisma } from "@/lib/prisma";
import { nanoid } from "nanoid";
import { addDays } from "date-fns";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { nickname, email, country, wish_text, category, visibility, reminder_days } = body;

    if (!nickname?.trim() || !email?.trim() || !wish_text?.trim()) {
      return NextResponse.json({ error: "Missing required fields." }, { status: 400 });
    }

    const public_id = nanoid(12);
    const reminder_date = addDays(new Date(), Number(reminder_days) || 365);

    const baseUrl = process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000";

    const session = await stripe.checkout.sessions.create({
      payment_method_types: ["card"],
      line_items: [
        {
          price_data: {
            currency: "usd",
            unit_amount: 99,
            product_data: {
              name: "Wish Candle",
              description: "Light a digital candle and make a wish. You'll be reminded in the future.",
              images: [],
            },
          },
          quantity: 1,
        },
      ],
      mode: "payment",
      success_url: `${baseUrl}/success?session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${baseUrl}/cancel`,
      customer_email: email,
      metadata: {
        public_id,
        nickname: nickname.slice(0, 40),
        email,
        country: country?.slice(0, 60) ?? "",
        wish_text: wish_text.slice(0, 500),
        wish_text_continued: wish_text.length > 500 ? wish_text.slice(500, 1000) : "",
        category: category ?? "general",
        visibility: visibility ?? "public",
        reminder_date: reminder_date.toISOString(),
      },
    });

    await prisma.wish.create({
      data: {
        public_id,
        nickname: nickname.trim(),
        email: email.trim(),
        country: country?.trim() ?? "",
        wish_text: wish_text.trim(),
        category: category ?? "general",
        visibility: visibility ?? "public",
        reminder_date,
        payment_status: "pending",
        stripe_session_id: session.id,
        status: "burning",
      },
    });

    return NextResponse.json({ url: session.url });
  } catch (err: unknown) {
    console.error("[create-checkout-session]", err);
    return NextResponse.json({ error: "Internal server error." }, { status: 500 });
  }
}
