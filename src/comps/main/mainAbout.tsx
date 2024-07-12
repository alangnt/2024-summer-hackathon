import Image from 'next/image';

export default function Main() {

    return (
        <main className="flex items-center flex-col grow pt-12 gap-8">
            <h2 className='text-2xl'>Meet the DataDruids Team !</h2>
            <section className='flex justify-between gap-12'>
                <article className='relative w-48 h-48'>
                    <Image src="/profiles/profile-conrad.jpg" alt='Profile picture of Conrad' fill objectFit='contain'></Image>
                    <div className='absolute top-48'>
                        <h5>Team Coordinator</h5>
                        <p>Conrad - @dashes_</p>
                    </div>
                </article>

                <article className='relative w-48 h-48'>
                    <Image src="/profiles/profile-milan.png" alt='Profile picture of Milan' fill objectFit='contain'></Image>
                    <div className='absolute top-48'>
                        <h5>Data Analyst</h5>
                        <p>Milan - @Milan</p>
                    </div>
                </article>

                <article className='relative w-48 h-48'>
                    <Image src="/profiles/profile-alan.jpg" alt='Profile picture of Alan' fill objectFit='contain'></Image>
                    <div className='absolute top-48'>
                        <h5>Web Developer</h5>
                        <p>Alan - @bxee</p>
                    </div>
                </article>
            </section>
        </main >
    )
}