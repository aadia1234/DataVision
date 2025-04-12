import { ChevronDown } from "lucide-react";

interface StatTestIndicatorProps {
  text: string;
  clicked: number;
  phaseNum: number; //0 is first step
}

const offset = 6;

export default function dropDown({
  text,
  clicked,
  phaseNum,
}: StatTestIndicatorProps) {
  return (
    <div>
      <p
        className={` flex flex-col w-full absolute left-10 clicked>2 ?"text-[17px]"
            :"loading"`}
      >
        {`${text} ${clicked}`}
      </p>
      <ChevronDown className="absolute left-15" size={50} />
      {/* {phaseNum > 0 && clicked > phaseNum && } */}
    </div>
  );
}
