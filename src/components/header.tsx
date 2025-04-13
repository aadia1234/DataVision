import { ChevronLeft } from "lucide-react";
import { Button } from "./ui/button";

export default function Header({ onClick }: { onClick: () => void }) {
    return (
        <div className="bg-gray-200 w-screen h-fit p-5">
            <Button onClick={onClick} variant="ghost" className="absolute left-5" size="icon">
                <ChevronLeft style={{ scale: 2 }} />
            </Button>
            <p className="text-4xl w-fit m-auto">Analysis</p>
        </div>
    );
}