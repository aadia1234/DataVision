import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
// interface dropDownProps {
//     text: string // title of block
//     clicked: number //state of processes
//     phaseNum: number //0 is first step
//     data: any // drop down info

//   }

export default function DropDown({
  text,
  clicked,
  phaseNum,
  data,
}: {
  text: string;
  clicked: number;
  phaseNum: number;
  data: any;
}) {    
  return (
    <div>
      <Accordion type="single" collapsible>
        <AccordionItem value="item-1">
          <AccordionTrigger disabled={ clicked <= phaseNum}>
            <div>
              <p
                className={`${clicked > phaseNum ? "" : "loading"} text-[17px]`}
              >
                {text}
              </p>
            </div>
          </AccordionTrigger>
          {clicked > phaseNum && <AccordionContent>{data}</AccordionContent>}
        </AccordionItem>
      </Accordion>
    </div>
  );
}
