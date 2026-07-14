import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { cn } from "@/lib/utils";
import { ThemeProvider } from "@/components/theme-provider";
import { Toaster } from "@/components/ui/sonner";
import { SiteNav } from "@/components/site-nav";

const geistSans = Geist({
  variable: "--font-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const siteUrl = process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000";

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: {
    default: "weave",
    template: "%s · weave",
  },
  description:
    "Tiny-model generative UI: a Needle-style tool call picks the layout, json-render draws it.",
  openGraph: {
    type: "website",
    siteName: "weave",
    title: "weave",
    description:
      "Tiny-model generative UI: a Needle-style tool call picks the layout, json-render draws it.",
    url: "/",
  },
  twitter: {
    card: "summary_large_image",
    title: "weave",
    description:
      "Tiny-model generative UI: a Needle-style tool call picks the layout, json-render draws it.",
  },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "white" },
    { media: "(prefers-color-scheme: dark)", color: "black" },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={cn(
        "h-full",
        "antialiased",
        geistSans.variable,
        geistMono.variable,
      )}
    >
      <body className="flex min-h-full flex-col">
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem={false}
          disableTransitionOnChange
        >
          <SiteNav />
          {children}
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  );
}
