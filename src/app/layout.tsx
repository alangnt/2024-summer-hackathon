import type { Metadata } from "next";
import "../styles/globals.css";

export const metadata: Metadata = {
  title: "Olympic Sages by DataDruids",
  description: "We are Team Data Druids, and we were tasked with predicting the winners of the 2024 Paris Olympics based on previous data. A significant portion of our data comes from the 2021 Tokyo Olympics. We analyzed historical data, visualized our results, and hypothesized how this data could be used to predict the outcomes of the 2024 Olympics. Our final step was to generate predictions for the 2024 Paris Olympics.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="overflow-x-hidden">
      <body className="flex flex-col items-center w-screen min-h-screen p-2 overflow-x-hidden text-white">{children}</body>
    </html>
  );
}
