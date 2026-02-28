import type { Metadata } from "next";
import { Inter, JetBrains_Mono, Patrick_Hand } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});
const logoFont = Patrick_Hand({
  subsets: ["latin"],
  variable: "--font-logo",
  weight: ["400"],
});

export const metadata: Metadata = {
  title: "La Corrigeuse - Correction automatique par IA",
  description: "Gagnez 90% de temps sur la correction de vos copies. IA double validation pour les professeurs de collège et lycée.",
  icons: {
    icon: "/favicon.svg?v=3",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} ${logoFont.variable} font-sans antialiased`}
      >
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
