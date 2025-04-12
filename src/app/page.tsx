"use client";

import Image from "next/image";
import { useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import { Upload, Loader2 } from "lucide-react";
import { toast } from "sonner";
import Analysis from "@/components/analysis";
import AnalysisHeader from "@/components/analysisHeader";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
<<<<<<< HEAD
  const [visuals, setVisuals] = useState<string[]>([]);
=======
  const [cleanResult, setCleanResult] = useState(null);
  const [designResult, setDesignResult] = useState<string | null>(null);
  const [visualizeResult, setVisualizeResult] = useState(null);
  const [analyzeResult, setAnalyzeResult] = useState(null);
  const [currentStep, setCurrentStep] = useState(0);
>>>>>>> main

  const getHypothesisVisuals = async () => {
    const url = "/api/hypothesis_visuals";
    const formData = new FormData();
    formData.append("file", file as Blob);
    
    try {
      const response = await fetch(url, {
        method: "POST",
        body: formData,
      });
  
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
  
      const result = await response.json();
      if (result.status === 200) {
        result.message.figures.forEach((base64String: string, index: number) => {
          const imgSrc = `data:image/png;base64,${base64String}`;
          console.log(`Image source for figure ${index + 1}:`, imgSrc);
          console.log(`Received figure ${index + 1}`);
        });
      }
      setVisuals(result.figures);
      console.log("Hypothesis Visuals:", result);
    } catch (error) {
      console.error("Error:", error);
      toast.error("Error getting hypothesis visuals");
    }
  };
  const cleanData = async () => {
    const url = "/api/data_cleaning";
    const formData = new FormData();
    formData.append("file", file as Blob);
    try {
      const response = await fetch(url, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const result = await response.json();
      setCleanResult(result);
      setCurrentStep(1);
      console.log("Cleaning Step:", result);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  const designProcedure = async () => {
    const url = "/api/design_procedure";
    const formData = new FormData();
    formData.append("file", file as Blob);
    try {
      const response = await fetch(url, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const result = await response.text();
      setDesignResult(result);
      setCurrentStep(2);
      console.log("Analysis Procedure:", result);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  // Run handleUpload when file changes
  useEffect(() => {
    if (file) {
      handleUpload();
    }
  }, [file]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];

      // Check if file is a CSV
      if (
        selectedFile.type === "text/csv" ||
        selectedFile.name.endsWith(".csv")
      ) {
        setFile(selectedFile);
        toast.success("CSV file selected", {
          description: selectedFile.name,
        });
      } else {
        // Show toast for invalid file type
        toast.error("Invalid file type", {
          description: "Please upload a CSV file",
        });

        // Reset the file input
        e.target.value = "";
      }
    }
  };

  const handleUpload = async () => {
    if (file && !isLoading) {
      setIsLoading(true);

      try {
        console.log("Processing file:", file.name);
        toast.success("Processing file", {
          description: file.name,
        });

        await cleanData();
        await designProcedure();
        await getHypothesisVisuals(); 

        // Implement your file upload/processing functionality here
        toast.success("File processed successfully", {
          description: file.name,
        });
      } catch (error) {
        toast.error("Error processing file", {
          description: "An unexpected error occurred",
        });
        console.error("Error:", error);
      } finally {
        setIsLoading(false);
      }
    } else if (!file) {
      console.log("No file selected");
      toast.error("No file selected", {
        description: "Please select a CSV file to upload",
      });
    }
  };

  return (
    <div className="bg-white flex items-center justify-center items-center w-full h-screen font-[family-name:var(--font-geist-sans)]">
      {!file && (
        <div className="flex flex-col gap-6 row-start-2 items-center justify-center w-3/4">
          <div className="flex flex-col gap-2 items-center justify-center text-center text-primary/80">
            <p className="text-5xl m-auto">DataVision</p>
            <i className="text-center w-full">
              Your intelligent assistant for exploring data, testing hypotheses,
              and generating visuals.
            </i>
          </div>

          <div className="flex flex-col w-full max-w-md gap-4">
            {isLoading ? (
              <div className="flex flex-col gap-2 items-center justify-center text-center text-sm text-primary/80">
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Processing your CSV file...
              </div>
            ) : (
              <div className="flex items-center justify-center gap-2">
                <Input
                  type="file"
                  id="fileUpload"
                  className="hidden"
                  accept=".csv,text/csv"
                  onChange={handleFileChange}
                  disabled={isLoading}
                />
                <label
                  htmlFor="fileUpload"
                  className={`text-lg text-center flex-col gap-2 m-auto px-20 py-10 border-2 border-dashed border-primary/50 cursor-pointer flex items-center rounded-md text-primary hover:bg-primary/10 ${
                    isLoading ? "opacity-50 pointer-events-none" : ""
                  }`}
                >
                  <Upload className="h-8 aspect-square" />
                  Upload CSV
                </label>
              </div>
            )}
          </div>
        </div>
      )}
      {file && (
        <div className="flex flex-col gap-2 w-2/3 h-screen items-center justify-center">
          <AnalysisHeader />
          <Analysis
<<<<<<< HEAD
            cleaned={true}
            designed={true}
            visualized={true}
            analyzed={true}
            visuals={visuals}
=======
            cleanResult={cleanResult}
            designResult={designResult}
            visualizeResult={visualizeResult}
            analyzeResult={analyzeResult}
            currentStep={currentStep}
>>>>>>> main
          />
        </div>
      )}
      {/*<div>
        {visuals.map((base64Str, index) => (
          <img
            key={`image-${index}`}
            src={`data:image/png;base64,${base64Str}`}
            alt={`Hypothesis Visual ${index}`}
          />
        ))}
      </div>*/}
      <footer className="bg-gray-200 w-full h-12 text-xs row-start-3 flex items-center justify-center fixed bottom-0">
        <a
          className="gap-2 text-gray-600 text-bold flex items-center hover:underline hover:underline-offset-4"
          href="https://github.com/aadia1234/DataVision"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/github-logo.svg"
            alt="Globe icon"
            width={12}
            height={12}
          />
          GitHub Repository
        </a>
      </footer>
    </div>
  );
}
