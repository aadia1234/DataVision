"use client";
import { useState } from "react";
import DropDown from "@/components/ui/dropDown";
import { Button } from "@/components/ui/button";
import { ChevronLeft } from "lucide-react";


//import { useRouter } from 'next/navigation'

interface AnalysisProps {
  cleaned: boolean;
  designed: boolean;
  visualized: boolean;
  analyzed: boolean;
}

export default function Analysis({
  cleaned,
  designed,
  visualized,
  analyzed,
}: AnalysisProps) {
  const [clicked, setClicked] = useState(0);
  const tempData = "tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData  "

  return (
    <div>
      <div className="w-screen h-screen bg-red-500">
        <div className="flex justify-center items-center h-12">
          <Button className="absolute left-2" variant="ghost" size="icon">
            <ChevronLeft style={{ scale: 2 }} />
          </Button>
          <p className="text-4xl mt-[-5px]">Analysis:</p>
        </div>
      </div>
    <main>
      
      <button
        className="bg-blue-500 text-white px-4 py-2 rounded"
        onClick={() => setClicked(clicked + 1)}>
        clean
      </button>
      

        {/* temporarly based on clicked variable need to add ani*/}
      <DropDown text="Cleaning data" clicked={clicked} phaseNum={0} data={tempData}/>
      {clicked>0 && <DropDown text="Finding Patterns" clicked={clicked} phaseNum={1} data={tempData}/>}
      {clicked>1 && <DropDown text="Running Statistical Tests" clicked={clicked} phaseNum={2} data={tempData}/>}
      {clicked>2 && <DropDown text="Found Data!" clicked={clicked} phaseNum={3} data={tempData}/>}
              
    </main>
    </div>
    
    
  );
}
