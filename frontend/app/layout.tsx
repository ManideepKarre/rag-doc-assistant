import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Document Assistant",
  description: "Upload documents and ask questions grounded in their content.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
