"use client";
import { useState } from "react";
import DropDown from "@/components/ui/dropDown";

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
    <main>
      
      <button
        className="bg-blue-500 text-white px-4 py-2 rounded"
        onClick={() => setClicked(clicked + 1)}>
        Click me
      </button>
        {/* temporarly based on clicked variable need to add ani*/}
      <DropDown text="Cleaning data" clicked={clicked} phaseNum={0} data={tempData}/>
      {clicked>0 && <DropDown text="Finding Patterns" clicked={clicked} phaseNum={1} data={tempData}/>}
      {clicked>1 && <DropDown text="Running Statistical Tests" clicked={clicked} phaseNum={2} data={tempData}/>}
      {clicked>2 && <DropDown text="Found Data!" clicked={clicked} phaseNum={3} data={tempData}/>}
              
    </main>
  );
}
