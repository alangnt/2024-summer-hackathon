import Link from 'next/link';

export default function Header() {

    return (
        <header className='flex'>
            <h1 className='text-3xl'>Olympic Sages</h1>

            <nav className='absolute right-4 flex gap-4'>
                <Link href="#">Home</Link>
                <Link href="#">About Us</Link>
            </nav>
        </header>
    )
}