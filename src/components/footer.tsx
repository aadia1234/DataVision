import Image from "next/image";

export default function Footer() {
    return (
        <footer className="bg-gray-200 w-full h-12 text-xs row-start-3 flex items-center justify-center fixed bottom-0">
            <a
                className="gap-2 text-gray-600 text-bold flex items-center hover:underline hover:underline-offset-4"
                href="https://github.com/aadia1234/DataVision"
                target="_blank"
                rel="noopener noreferrer"
            >
                <Image
                    aria-hidden
                    src="/github-logo.svg"
                    alt="Globe icon"
                    width={12}
                    height={12}
                />
                GitHub Repository
            </a>
        </footer>
    );
}