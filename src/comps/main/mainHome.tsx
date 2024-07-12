import * as Graphs from "@/comps/graphs";

export default function Home() {

    return (
        <main className="flex flex-col grow items-center">
            <h3>Olympic Sages is a website made by three people, Milan, Conrad and Alan. It features some predictions for the 2024 Paris Summer Olympic winners</h3>

            {/* Section containing the four pies with TOT, GOLD, SILVER and BRONZE Counts */}
            <section className="flex w-screen flex-wrap">

                <Graphs.MostTotalMedalsPie />

                <Graphs.MostGoldMedalsPie />

                <Graphs.MostSilverMedalsPie />

                <Graphs.MostBronzeMedalsPie />

            </section>

            <h3>First, we tried to predict the winners using the Linear Regression method</h3>

            <Graphs.LinearRegressionBar />

            <Graphs.LinearRegressionScatter />

            <h3>Now, let's try again using the Random Forest method, -- EXPLAIN WHAT THIS IS</h3>

            <Graphs.RandomForestBar />

            <Graphs.RandomForestScatter />
        </main>
    )
}