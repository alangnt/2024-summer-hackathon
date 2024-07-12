import Header from "@/comps/header/headerHome";
import Main from "@/comps/main/mainHome";
import Footer from "@/comps/footer";
import Image from "next/image";

export default function Home() {
  return (
    <>
      <Header />

      <Main />

      <Image src="/medals/olympics.png" alt="Olympic medals pixelated" width={75} height={75}></Image>

      <Footer />
    </>
  );
}
