import * as Graphs from "@/comps/graphs";
import Image from "next/image";

export default function Home() {

    return (
        <main className="flex flex-col grow items-center gap-12">
            <h3 className="text-center">We wanted to understand how we could use data from previous olympics to predict future olympics. This section showcases some analysis that we conducted using a dataset from the 2021 Tokyo Olympics.</h3>

            <Image src="/medals/olympics.png" alt="Olympic medals pixelated" width={75} height={75}></Image>

            {/* Section containing the four pies with TOT, GOLD, SILVER and BRONZE Counts */}
            <section className="flex justify-center flex-wrap w-screen gap-4">
                <p className="w-3/4 text-center">We have created four interactive pie charts showcasing the top ten countries in various medal categories. These charts allow you to see the number of medals each country won by hovering over the respective segments. Additionally, you can toggle countries on and off to observe how this affects the overall medal distribution.</p>
                <article className="flex justify-center flex-wrap">
                    <Graphs.MostTotalMedalsPie />
                    <Graphs.MostGoldMedalsPie />
                    <Graphs.MostSilverMedalsPie />
                    <Graphs.MostBronzeMedalsPie />
                </article>
            </section>

            <Image src="/medals/gold.png" alt="Gold medal pixelated" width={75} height={75}></Image>

            <section className="flex justify-center flex-wrap w-screen gap-4">
                <p className="w-3/4 text-center">We have created a radar chart displaying the top five countries with the most medals. This chart illustrates the distribution of gold, silver, and bronze medals relative to the total number of medals each country won. You can toggle each category on and off to gain a clearer understanding of how these countries compare to one another.</p>
                <Graphs.DistributionMedalsRadar />
                <Graphs.MedalsPerCountryBar />
            </section>

            <Image src="/medals/silver.png" alt="Silver medal pixelated" width={75} height={75}></Image>

            <section>
                <h3 className="text-center">First, we tried to predict the winners using the Linear Regression method</h3>

                <article className="flex justify-center w-screen flex-wrap">
                    <Graphs.LinearRegressionBar />
                    <Graphs.LinearRegressionScatter />
                </article>
            </section>

            <Image src="/medals/bronze.png" alt="Bronze medal pixelated" width={75} height={75}></Image>

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