import {
    Accordion,
    AccordionContent,
    AccordionItem,
    AccordionTrigger,
  } from "@/components/ui/accordion"
// interface dropDownProps {
//     text: string // title of block
//     clicked: number //state of processes
//     phaseNum: number //0 is first step
//     data: any // drop down info

//   }

export default function DropDown({text, clicked, phaseNum, data}: {
    text: string
    clicked: number 
    phaseNum: number
    data: any }
) {
    return(
     <div>
        <Accordion type="single" collapsible>
            <AccordionItem value="item-1">
              <AccordionTrigger>
                <div>
                <p className={clicked>phaseNum ?"text-[17px] absolute left-5":"absolute left-5 loading"}>
                    {text}
                </p>
                </div>
              </AccordionTrigger>
              {clicked > phaseNum &&
              <AccordionContent>
                {data}
              </AccordionContent>
              }
            </AccordionItem>
          </Accordion>
    </div>)
}
