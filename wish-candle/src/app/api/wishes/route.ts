import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const sessionId = searchParams.get("session_id");
  const publicId = searchParams.get("public_id");

  if (sessionId) {
    const wish = await prisma.wish.findUnique({
      where: { stripe_session_id: sessionId },
      select: { public_id: true, payment_status: true },
    });
    if (!wish) return NextResponse.json({ error: "Not found" }, { status: 404 });
    return NextResponse.json(wish);
  }

  if (publicId) {
    const wish = await prisma.wish.findUnique({
      where: { public_id: publicId },
    });
    if (!wish) return NextResponse.json({ error: "Not found" }, { status: 404 });
    return NextResponse.json(wish);
  }

  return NextResponse.json({ error: "Missing parameter" }, { status: 400 });
}

export async function PATCH(req: NextRequest) {
  try {
    const { public_id, status } = await req.json();

    if (!public_id || !["fulfilled", "not_yet"].includes(status)) {
      return NextResponse.json({ error: "Invalid request." }, { status: 400 });
    }

    const wish = await prisma.wish.findUnique({ where: { public_id } });
    if (!wish || wish.payment_status !== "paid") {
      return NextResponse.json({ error: "Wish not found." }, { status: 404 });
    }

    const updated = await prisma.wish.update({
      where: { public_id },
      data: { status, responded_at: new Date() },
    });

    return NextResponse.json(updated);
  } catch (err) {
    console.error("[PATCH /api/wishes]", err);
    return NextResponse.json({ error: "Internal server error." }, { status: 500 });
  }
}
