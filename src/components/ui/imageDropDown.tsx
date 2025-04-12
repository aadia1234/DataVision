import {
    Accordion,
    AccordionContent,
    AccordionItem,
    AccordionTrigger,
  } from "@/components/ui/accordion";
  
  export default function ImageDropDown({
    text,
    clicked,
    phaseNum,
    data,
    images,
  }: {
    text: string;
    clicked: number;
    phaseNum: number;
    data: any;
    images: string[];
  }) {
    return (
      <div>
        <Accordion type="single" collapsible>
          <AccordionItem value="item-1">
            <AccordionTrigger>
              <div>
                <p className={`${clicked > phaseNum ? "" : "loading"} text-[17px]`}>
                  {text}
                </p>
              </div>
            </AccordionTrigger>
            {clicked > phaseNum && (
              <AccordionContent>
                <div className="flex flex-col gap-4">
                  {data}
                  {images.map((base64Str, index) => (
                    <img
                      key={`image-${index}`}
                      src={`data:image/png;base64,${base64Str}`}
                      alt={`Hypothesis Visual ${index + 1}`}
                      className="w-full max-w-xl rounded-md border border-gray-300 shadow"
                    />
                  ))}
                </div>
              </AccordionContent>
            )}
          </AccordionItem>
        </Accordion>
      </div>
    );
  }
  