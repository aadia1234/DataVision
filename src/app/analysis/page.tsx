"use client";

import { useState, useEffect } from "react";
import DropDown from "@/components/drop-down";
import { Button } from "@/components/ui/button";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { ChevronLeft } from "lucide-react";
import { useSearchParams } from "next/navigation";
import Footer from "@/components/footer";
import Header from "@/components/header";

export default function Analysis() {
    const tempData = "tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData  ";

    const [cleanSummary, setCleanSummary] = useState<string | null>(null);
    const [codeForCleaning, setCodeForCleaning] = useState<string | null>(null);

    const searchParams = useSearchParams();
    const [analysis, setAnalysis] = useState<Analysis>();

    const id = searchParams.get("id");

    useEffect(() => {
        const load = async () => {
            const cleanSummary = await cleanData();
            const designRes = await designProcedure();
            await hypothesisTest(designRes ?? null);
            await summarize(cleanSummary ?? null, designRes ?? null, "p values go here");
        };

        load();
    }, []);

    const cleanData = async () => {
        const url = "/api/data_cleaning";

        try {
            const response = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ id }),
            });

            console.log(response);

            if (!response.ok) {
                throw new Error("Network response was not ok");
            }

            const result = await response.json();
            console.log("TEST");
            setAnalysis((prev) => ({ ...prev, cleanResult: result })); return result.summary;
        } catch (error) {
            console.error("Error:", error);
        }
    };

    const designProcedure = async () => {
        const url = "/api/design_procedure";

        console.log("DESIGN PROCEDURE: ", id);

        try {
            const response = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ id }),
            });

            if (!response.ok) {
                throw new Error("Network response was not ok");
            }

            const result = await response.text();
            setAnalysis((prev) => ({ ...prev, designResult: result }));
            console.log("Analysis Procedure:", result);
            return result;
        } catch (error) {
            console.error("Error:", error);
        }
    };

    // fix endpoint
    const hypothesisTest = async (designResult: string | null) => {
        const url = "/api/hypothesis_test";
        console.log("DESIGN RESULT: ", designResult);
        try {
            const formData = new FormData();

            if (designResult) {
                formData.append("designResult", designResult);
            } else {
                console.error("Design result is missing");
                return;
            }

            const response = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: formData,
            });

            if (!response.ok) {
                throw new Error("Network response was not ok");
            }

            const result = await response.json();
            console.log("Raw response:", result);
            setAnalysis({ ...analysis, hypothesisTestingResult: result });
        } catch (error) {
            console.error("Error:", error);
        }
    };

    const summarize = async (
        cleaning_summary: string | null,
        potential_relationships: string | null,
        p_values_summary: string | null
    ) => {
        const url = "/api/summarize";
        const data = {
            id: id,
            cleaning_summary,
            potential_relationships,
            p_values_summary,
        };

        try {
            const response = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data),
            });
        } catch (error) {
            console.error("Error:", error);
        }
    };

    const downloadCSV = () => {
        if (analysis?.cleanResult && analysis.cleanResult.csv) {
            const blob = new Blob([analysis.cleanResult.csv], { type: "text/csv" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "cleaned_data.csv"; // Name of the downloaded file
            a.click();
            URL.revokeObjectURL(url); // Clean up the URL object
        } else {
            console.error("No CSV data available for download");
        }
    };

    useEffect(() => {
        if (analysis?.cleanResult) {
            setCleanSummary(analysis.cleanResult["summary"]);
            setCodeForCleaning(analysis.cleanResult["code"]);
        } else {
            setCleanSummary(null);
            setCodeForCleaning(null);
        }
        console.log("Summary: ", cleanSummary);
    }, [analysis?.cleanResult]);

    const CodeDisplay = ({ code }: { code: string | null }) => {
        if (!code) {
            return <p>No code available</p>;
        }
        return <SyntaxHighlighter language="javascript">{code}</SyntaxHighlighter>;
    };

    const CleaningStepData = (
        <div className="flex flex-col gap-2">
            <span className="font-bold">Summary:</span>
            <p>{cleanSummary}</p>
            <span className="font-bold">Code:</span>
            <CodeDisplay code={codeForCleaning} />
            <Button onClick={downloadCSV} className="mt-4">
                Download CSV
            </Button>
        </div>
    );

    const DesignStepData = (
        <div className="flex flex-col gap-2">
            <span className="font-bold">Proposed Procedure:</span>
            <p>{analysis?.designResult}</p>
        </div>
    );

    return (
        <div className="bg-white flex flex-col w-full h-screen font-[family-name:var(--font-geist-sans)]">
            <Header onClick={() => window.history.back()} />
            <div className="w-full">
                {/* <button
        className="bg-blue-500 text-white px-4 py-2 rounded"
        onClick={() => setClicked(clicked + 1)}
      >
        clean
      </button> */}

                {/* temporarly based on clicked variable need to add ani*/}
                <DropDown
                    text="Cleaning Data"
                    view={CleaningStepData}
                />
                {analysis?.designResult && (
                    <DropDown
                        text="Designing Analysis Procedure"
                        view={DesignStepData}
                    />
                )}
                {/* {analysis?.hypothesisTestingResult && (
                <DropDown
                    text="Running Statistical Tests"
                    view={hypothesisTestingResult}
                />
            )} */}
                {/* {analysis?.cleanResult && (
                <DropDown
                    text="Found Data!"
                    view={tempData}
                />
            )} */}

            </div>
            <Footer />
        </div>
    );
}
