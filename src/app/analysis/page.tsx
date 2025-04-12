"use client";
import Analysis from "@/components/analysis";
import Image from "next/image";
import { ChevronLeft } from "lucide-react";
import { ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState } from 'react'
import DropDown from "@/components/ui/dropDown";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"


//import { useRouter } from 'next/navigation'

export default function AnalysisPage() {
 
  const [clicked, setClicked] = useState(0)

  return (
    
      <main>
        <Analysis cleaned={true} designed={true} visualized={true} analyzed={true} />
      </main>
      
  );
}
