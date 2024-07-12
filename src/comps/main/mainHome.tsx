import * as Graphs from "@/comps/graphs";
import Image from "next/image";

export default function Home() {

    return (
        <main className="flex flex-col grow items-center gap-12">
            <h3 className="text-center">Olympic Sages is a website made by three people, Milan, Conrad and Alan. It features some predictions for the 2024 Paris Summer Olympic winners</h3>

            {/* Section containing the four pies with TOT, GOLD, SILVER and BRONZE Counts */}
            <section className="flex justify-center w-screen flex-wrap">
                <Graphs.MostTotalMedalsPie />
                <Graphs.MostGoldMedalsPie />
                <Graphs.MostSilverMedalsPie />
                <Graphs.MostBronzeMedalsPie />
            </section>

            <Graphs.DistributionMedalsRadar />

            <Graphs.MedalsPerCountryBar />

            <Image src="/medals/gold.png" alt="Gold medal pixelated" width={75} height={75}></Image>

            <section>
                <h3 className="text-center">First, we tried to predict the winners using the Linear Regression method</h3>

                <article className="flex justify-center w-screen flex-wrap">
                    <Graphs.LinearRegressionBar />
                    <Graphs.LinearRegressionScatter />
                </article>
            </section>

            <Image src="/medals/silver.png" alt="Silver medal pixelated" width={75} height={75}></Image>

            <section className="text-center">
                <h3>Now, let's try again using the Random Forest method, -- EXPLAIN WHAT THIS IS</h3>

                <article className="flex justify-center w-screen flex-wrap">
                    <Graphs.RandomForestBar />
                    <Graphs.RandomForestScatter />
                </article>
            </section>
        </main >
    )
}