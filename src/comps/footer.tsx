import Image from 'next/image';

export default function Footer() {

    return (
        <footer className="flex flex-col items-center w-full">
            <nav className='flex justify-between gap-2'>
                <article className='flex flex-col gap-2'>
                    <Image src="/socials/linkedin.png" alt="LinkedIn logo" width={16} height={16}></Image>
                    <Image src="/socials/instagram.png" alt="Instagram logo" width={16} height={16}></Image>
                </article>

                <article className='flex flex-col gap-2'>
                    <Image src="/socials/linkedin.png" alt="LinkedIn logo" width={16} height={16}></Image>
                    <Image src="/socials/github.png" alt="GitHub logo" width={16} height={16}></Image>
                </article>

                <article className='flex flex-col gap-2'>
                    <Image src="/socials/linkedin.png" alt="LinkedIn logo" width={16} height={16}></Image>
                    <Image src="/socials/instagram.png" alt="Instagram logo" width={16} height={16}></Image>
                </article>
            </nav>

            <p>&copy; 2024 Olympic Sages | Codédex 2024 Summer Hackathon.</p>
        </footer>
    )
}