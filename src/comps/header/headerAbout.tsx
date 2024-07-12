import Link from 'next/link';

export default function Header() {

    return (
        <header className='flex mb-4'>
            <h1 className='text-5xl'>About Us</h1>

            <nav className='absolute right-4 flex gap-4 text-2xl'>
                <Link href="/">Home</Link>
                <Link href="/about-us">About Us</Link>
            </nav>
        </header>
    )
}