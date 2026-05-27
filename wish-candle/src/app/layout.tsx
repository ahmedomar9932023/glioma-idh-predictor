import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Wish Candle — Light a Candle, Make a Wish",
  description:
    "Light a digital candle for $0.99, write your deepest wish, and receive it back one year later to see if it came true.",
  keywords: ["wish", "candle", "magic", "dream", "hope"],
  openGraph: {
    title: "Wish Candle",
    description: "Light a candle. Make a wish. See if it came true.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className="antialiased min-h-screen relative">
        {children}
      </body>
    </html>
  );
}
