import Link from 'next/link';

export default function Header() {

    return (
        <header className='flex mb-4'>
            <h1 className='text-5xl'>About Us</h1>

            <nav className='absolute right-4 flex gap-4 text-2xl'>
                <Link href="/" className='hover:scale-110'>Home</Link>
                <Link href="/about-us" className='hover:scale-110'>About Us</Link>
            </nav>
        </header>
    )
}