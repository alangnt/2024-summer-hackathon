import Image from 'next/image';
import Link from 'next/link';

export default function Footer() {

    return (
        <footer className="flex flex-col items-center w-full gap-12 pt-12 pb-2">
            <p>Follow the Links to keep in touch !</p>

            <nav className='flex justify-between gap-2 w-1/2'>
                <article className='flex flex-col gap-2'>
                    <h5 className='flex h-4 justify-center items-center'>Conrad</h5>
                    <div className='flex gap-4'>
                        <div className='flex gap-2 h-4 justify-center items-center'>
                            <Image src="/socials/linkedin.png" alt="LinkedIn logo" width={16} height={16}></Image>
                            <Link href="https://www.linkedin.com/in/conradwyrick" target="_blank">LinkedIn</Link>
                        </div>
                        <div className='flex gap-2 h-4 justify-center items-center'>
                            <Image src="/socials/instagram.png" alt="Instagram logo" width={16} height={16}></Image>
                            <Link href="https://www.instagram.com/con_radical/" target="_blank">Instagram</Link>
                        </div>
                    </div>
                </article>

                <article className='flex flex-col gap-2'>
                    <h5 className='flex h-4 justify-center items-center'>Milan</h5>
                    <div className='flex gap-4'>
                        <div className='flex gap-2 h-4 justify-center items-center'>
                            <Image src="/socials/linkedin.png" alt="LinkedIn logo" width={16} height={16}></Image>
                            <Link href="https://www.linkedin.com/in/milan-grujicic-20ba05110/" target="_blank">LinkedIn</Link>
                        </div>
                        <div className='flex gap-2 h-4 justify-center items-center'>
                            <Image src="/socials/github.png" alt="GitHub logo" width={16} height={16}></Image>
                            <Link href="https://github.com/MilanGrujicic" target="_blank">GitHub</Link>
                        </div>
                    </div>
                </article>

                <article className='flex flex-col gap-2'>
                    <h5 className='flex h-4 justify-center items-center'>Alan</h5>
                    <div className='flex gap-4'>
                        <div className='flex gap-2 h-4 justify-center items-center'>
                            <Image src="/socials/linkedin.png" alt="LinkedIn logo" width={16} height={16}></Image>
                            <Link href="https://www.linkedin.com/in/alan-geirnaert/" target="_blank">LinkedIn</Link>
                        </div>
                        <div className='flex gap-2 h-4 justify-center items-center'>
                            <Image src="/socials/instagram.png" alt="Instagram logo" width={16} height={16}></Image>
                            <Link href="https://www.instagram.com/gnt_alan/" target="_blank">Instagram</Link>
                        </div>
                    </div>
                </article>
            </nav>

            <p>&copy; 2024 Olympic Sages | Codédex 2024 Summer Hackathon.</p>
        </footer>
    )
}