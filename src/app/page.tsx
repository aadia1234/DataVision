"use client";

import Image from "next/image";
import { useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import { Upload, Loader2 } from "lucide-react";
import { toast } from "sonner";

export default function Home() {
  const [searchQuery, setSearchQuery] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [demoText, setDemoText] = useState("Demo text");

  const fetchData = async () => {
    const res = await fetch("/api/hello");
    const text = await res.text();
    console.log("TEXT: " + text);
    setDemoText(text);
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

        await fetchData();

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
    <div className="bg-white grid grid-rows-[56px_1fr_56px] items-center justify-items-center min-h-screen font-[family-name:var(--font-geist-sans)]">
      <main className="flex flex-col gap-8 row-start-2 items-center sm:items-start">
        <p className="text-5xl m-auto">DataVision</p>
        <p>{demoText}</p>

        <div className="flex flex-col w-full max-w-md gap-4">
          <div className="flex items-center gap-2">
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
              className={`m-auto cursor-pointer flex items-center px-4 py-2 rounded-md bg-primary text-primary-foreground hover:bg-primary/90 ${
                isLoading ? "opacity-50 pointer-events-none" : ""
              }`}
            >
              <Upload className="h-4 w-4 mr-2" />
              Upload CSV
            </label>
          </div>

          {isLoading && (
            <div className="flex items-center justify-center text-center text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Processing your CSV file...
            </div>
          )}
        </div>
      </main>
      <footer className="bg-gray-200 w-full h-14 row-start-3 flex items-center justify-center">
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
            width={16}
            height={16}
          />
          GitHub Repository
        </a>
      </footer>
    </div>
  );
}
