import { Button } from "@/components/ui/button";
import { ChevronLeft } from "lucide-react";

export default function AnalysisHeader() {
  return (
    <div>
      <div className="w-screen h-10 fixed top-0 left-0 pt-6">
        <div className="flex justify-center items-center h-10">
          <Button variant="ghost" className="absolute left-5" size="icon">
            <ChevronLeft style={{ scale: 2 }} />
          </Button>
          <p className="text-4xl mt-[-5px]">Analysis:</p>
        </div>
      </div>
    </div>
  );
}
