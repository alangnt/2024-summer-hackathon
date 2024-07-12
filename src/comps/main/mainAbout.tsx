import Image from 'next/image';

export default function Main() {

    return (
        <main className="flex items-center flex-col grow">
            <h2>Meet the DataDruids Team !</h2>
            <section className='flex justify-between'>
                <article>
                    <Image src="/profiles/profile-alan.jpg" alt='Profile picture of Alan' width={250} height={250}></Image>
                    <p>Alan - @bxee</p>
                </article>

                <article>
                    <Image src="/profiles/profile-alan.jpg" alt='Profile picture of Alan' width={250} height={250}></Image>
                    <p>Conrad - @dashes_</p>
                </article>

                <article>
                    <Image src="/profiles/profile-alan.jpg" alt='Profile picture of Alan' width={250} height={250}></Image>
                    <p>Milan - @Milan</p>
                </article>
            </section>
        </main>
    )
}