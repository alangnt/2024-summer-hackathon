import * as Graphs from "@/comps/graphs";
import Image from "next/image";

export default function Home() {

    return (
        <main className="flex flex-col grow items-center gap-12">
            <div className="flex flex-col items-center text-center w-3/4">
                <h3>Preliminary Analysis</h3>
                <p className="text-center">We wanted to understand how we could use data from previous olympics to predict future olympics. This section showcases some analysis that we conducted using a dataset from the 2021 Tokyo Olympics.</p>
            </div>

            <Image src="/medals/olympics.png" alt="Olympic medals pixelated" width={75} height={75}></Image>

            {/* Section containing the four pies with TOT, GOLD, SILVER and BRONZE Counts */}
            <section className="flex justify-center flex-wrap w-screen gap-4">
                <p className="w-3/4 text-center">We have created four interactive pie charts showcasing the top ten countries in various medal categories. These charts allow you to see the number of medals each country won by hovering over the respective segments. Additionally, you can toggle countries on and off to observe how this affects the overall medal distribution.</p>
                <article className="flex flex-col items-center">
                    <div className="flex small-screens">
                        <Graphs.MostTotalMedalsPie />
                        <Graphs.MostGoldMedalsPie />
                    </div>
                    <div className="flex small-screens">
                        <Graphs.MostSilverMedalsPie />
                        <Graphs.MostBronzeMedalsPie />
                    </div>
                </article>
            </section>

            <Image src="/medals/gold.png" alt="Gold medal pixelated" width={75} height={75}></Image>

            <section className="flex justify-center flex-wrap w-screen gap-4">
                <p className="w-3/4 text-center">We have created a radar chart displaying the top five countries with the most medals. This chart illustrates the distribution of gold, silver, and bronze medals relative to the total number of medals each country won. You can toggle each category on and off to gain a clearer understanding of how these countries compare to one another.</p>
                <Graphs.DistributionMedalsRadar />
                <Graphs.MedalsPerCountryBar />
            </section>

            <section className="flex justify-center flex-wrap w-screen gap-4">
                <p className="w-3/4 text-center">We have created a stacked bar chart featuring all the countries that participated in the 2021 Tokyo Olympics, displaying their respective gold, silver, and bronze medals. You can toggle each medal category to see how the countries compare to one another.</p>
                <Graphs.MedalsPerCountryBar />
            </section>

            <Image src="/medals/silver.png" alt="Silver medal pixelated" width={75} height={75}></Image>

            <section className="flex justify-center flex-wrap w-screen gap-4">
                <p className="w-3/4 text-center">This linear regression model displays the top 10 countries with the highest predicted total medals based on the number of athletes they sent to the 2021 Tokyo Olympics. The model's mean absolute error is 2.8, and the R-squared value is -0.8, indicating that the model did not perform well. Improvements will be needed moving forward. This model is interactive and you can see more information when hovering over a section of the models.</p>
                <article className="flex justify-center w-screen flex-wrap">
                    <Graphs.LinearRegressionBar />
                    <Graphs.LinearRegressionScatter />
                </article>
            </section>

            <Image src="/medals/bronze.png" alt="Bronze medal pixelated" width={75} height={75}></Image>

            <section className="flex justify-center flex-wrap w-screen gap-4">
                <p className="w-3/4 text-center">This random forest regressor model also displays the top 10 countries with the highest predicted total medals, but now is based on the number of athletes and coaches they sent to the 2021 Tokyo Olympics. Several improvements were made from the previous model. The inclusion of coaches data provides a more comprehensive view, and Grid Search was used to fine-tune the model. The model's mean absolute error is 0.5, and the R-squared value is 0.9, indicating that it performs well. This model is interactive and you can see more information when hovering over a section of the models.</p>
                <article className="flex justify-center w-screen flex-wrap">
                    <Graphs.RandomForestBar />
                    <Graphs.RandomForestScatter />
                </article>
            </section>
        </main >
    )
}