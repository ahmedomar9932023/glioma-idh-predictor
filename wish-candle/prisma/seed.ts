import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

async function main() {
  const wishes = [
    {
      nickname: "StarGazer",
      email: "demo1@example.com",
      country: "US",
      wish_text: "I wish to travel the world and find peace in every corner of the earth.",
      category: "adventure",
      visibility: "public",
      reminder_date: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000),
      payment_status: "paid",
      status: "burning",
    },
    {
      nickname: "MoonDreamer",
      email: "demo2@example.com",
      country: "GB",
      wish_text: "May my family stay healthy and happy through every season.",
      category: "health",
      visibility: "public",
      reminder_date: new Date(Date.now() + 180 * 24 * 60 * 60 * 1000),
      payment_status: "paid",
      status: "burning",
    },
    {
      nickname: "FireKeeper",
      email: "demo3@example.com",
      country: "JP",
      wish_text: "I wish to finish my novel and share it with the world.",
      category: "creative",
      visibility: "public",
      reminder_date: new Date(Date.now() + 90 * 24 * 60 * 60 * 1000),
      payment_status: "paid",
      status: "fulfilled",
    },
    {
      nickname: "CandleWarden",
      email: "demo4@example.com",
      country: "FR",
      wish_text: "To find love that lasts a lifetime, warm and true.",
      category: "love",
      visibility: "public",
      reminder_date: new Date(Date.now() + 270 * 24 * 60 * 60 * 1000),
      payment_status: "paid",
      status: "burning",
    },
    {
      nickname: "ShadowSeeker",
      email: "demo5@example.com",
      country: "DE",
      wish_text: "That I may find the courage to take the leap and start my own business.",
      category: "career",
      visibility: "public",
      reminder_date: new Date(Date.now() + 150 * 24 * 60 * 60 * 1000),
      payment_status: "paid",
      status: "burning",
    },
  ];

  for (const wish of wishes) {
    await prisma.wish.create({ data: wish });
  }

  console.log("✨ Database seeded with", wishes.length, "wishes.");
}

main()
  .catch(console.error)
  .finally(() => prisma.$disconnect());
