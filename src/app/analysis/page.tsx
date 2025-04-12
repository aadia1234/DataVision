"use client";
import Image from "next/image";
import { ChevronLeft } from "lucide-react";
import { ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { s } from "framer-motion/client";
import dropDown from "@/components/ui/dropDown";
import StatTestIndicator from "@/components/ui/dropDown";

//import { useRouter } from 'next/navigation'

export default function Analysis() {
  const [clicked, setClicked] = useState(0);
  const [showDiv, setShowDiv] = useState(false);

  return (
    <div className="w-screen h-screen">
      <div className="flex justify-center items-center h-12">
        {/* Top-left element */}
        <Button className="absolute left-2" variant="ghost" size="icon">
          <ChevronLeft style={{ scale: 2 }} />
        </Button>
        <p className="text-4xl mt-[-5px]">Analysis:</p>
      </div>
      <main>
        <button
          className="bg-blue-500 text-white px-4 py-2 rounded"
          onClick={() => setClicked(clicked + 1)}
        >
          Click me
        </button>
        <div className="absolute mt-10">
          <StatTestIndicator
            text="testttdsfasdfasdfttt"
            clicked={clicked}
            phaseNum={1}
          />
        </div>

        {/* {clicked>0 && (<button
        className=" flex flex-col bg-blue-500 text-white px-4 py-2 rounded"
        onClick={() => setClicked(clicked + 1)}
      >
        Click me
      </button>)} */}

        {/* <AnimatePresence>
        {showDiv && (
          <motion.div
            initial={{y: -35, opacity: 0}}
            animate={{ y: 0, opacity: 1}}
            transition={{ duration: 0.4 }}
            className="bg-green-100 p-4 rounded shadow"
          >
            🎉 I'm animated!
          </motion.div>
        )}
      </AnimatePresence> */}
        {/* temporarly based on clicked variable need to add ani*/}

        <div className="proccessing-data">
          {/* 1 Cleaning Data*/}
          <p
            className={
              clicked > 0
                ? "text-[17px] absolute left-10"
                : "absolute left-10 loading"
            }
          >
            Loading data
          </p>

          {/* 2 Finding patterns */}
          {clicked > 0 && (
            <div>
              <p
                className={
                  clicked > 1
                    ? "text-[17px] absolute left-10 mt-6"
                    : "absolute left-10 loading mt-6"
                }
              >
                Finding patterns
              </p>
              <ChevronDown className="absolute right-10" size={25} />
            </div>
          )}

          {/* 3 Running Statistical Tests*/}
          {clicked > 1 && (
            <div>
              <p
                className={
                  clicked > 2
                    ? "text-[17px] absolute left-10 mt-12"
                    : "absolute left-10 loading mt-12"
                }
              >
                Running Statistical Tests
              </p>
              <ChevronDown className="absolute right-10 mt-6" size={25} />
            </div>
          )}

          {/* Found Data!*/}
          {clicked > 2 && (
            <div>
              <p className="text-[17px] absolute left-10 mt-18">Found Data!</p>
              <ChevronDown className="absolute right-10 mt-12" size={25} />
            </div>
          )}
        </div>
      </main>
      <footer className="bg-gray-200 w-full h-14 row-start-3 flex items-center justify-center absolute bottom-0">
        <a
          className="gap-2 text-gray-600 flex items-center hover:underline hover:underline-offset-4"
          href="https://www.aadiananddeveloper05.com"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/github-logo.svg"
            alt="Globe icon"
            width={16}
            height={16}
          />
          GitHub Repository
        </a>
      </footer>
    </div>
  );
}
