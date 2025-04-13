"use client";

import { useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import { Upload, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { useRouter, useSearchParams } from "next/navigation";
import Footer from "@/components/footer";

export default function Home() {
  const [file, setFile] = useState<File>();
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();
  const searchParams = useSearchParams();


  // Run handleUpload when file changes
  useEffect(() => {
    if (file) {
      handleUpload();
    }
  }, [file]);

  const uploadFile = async () => {
    const url = "/api/upload";
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

      return result.id;
    } catch (error) {
      console.error("Error:", error);
    }
  }

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
        const id = await uploadFile();
        const params = new URLSearchParams(searchParams);
        params.set("id", id);
        router.push("/analysis?" + params.toString());
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
    <div className="bg-white flex justify-center items-center w-full h-screen font-[family-name:var(--font-geist-sans)]">
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
                  className={`text-lg text-center flex-col gap-2 m-auto px-20 py-10 border-2 border-dashed border-primary/50 cursor-pointer flex items-center rounded-md text-primary hover:bg-primary/10 ${isLoading ? "opacity-50 pointer-events-none" : ""
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
      <Footer />
    </div>
  );
}
