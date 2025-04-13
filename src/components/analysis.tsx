"use client";

import { useState, useEffect, useRef } from "react";
import DropDown from "@/components/drop-down";
import { Button } from "@/components/ui/button";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import Chat from "./chat";

interface AnalysisProps {
  cleanResult: Record<string, any> | null;
  designResult: string | null;
  hypothesisTestingResult: any | null;
  analyzeResult: string | null;
  currentStep: number;
}

interface Message {
  id: string;
  content: string;
  role: "user" | "assistant";
  timestamp: Date;
}

export default function Analysis({
  cleanResult,
  designResult,
  hypothesisTestingResult,
  analyzeResult,
  currentStep,
}: AnalysisProps) {
  const tempData =
    "tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData tempData  ";
  const [cleanSummary, setCleanSummary] = useState<string | null>(null);
  const [codeForCleaning, setCodeForCleaning] = useState<string | null>(null);

  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      content:
        "Hello! I'm your data analysis assistant. Ask me anything about your analysis.",
      role: "assistant",
      timestamp: new Date(),
    },
  ]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const downloadCSV = () => {
    if (cleanResult && cleanResult.csv) {
      const blob = new Blob([cleanResult.csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "cleaned_data.csv";
      a.click();
      URL.revokeObjectURL(url);
    } else {
      console.error("No CSV data available for download");
    }
  };

  useEffect(() => {
    if (cleanResult) {
      setCleanSummary(cleanResult["summary"]);
      setCodeForCleaning(cleanResult["code"]);
    } else {
      setCleanSummary(null);
      setCodeForCleaning(null);
    }
    console.log("Summary: ", cleanSummary);
  }, [cleanResult]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async (message: string): Promise<string> => {
    if (!message.trim()) return Promise.reject("Message cannot be empty.");

    const newUserMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      role: "user",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, newUserMessage]);
    setInputMessage("");
    setIsLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: message,
          currentStep: currentStep,
          history: messages.map((msg) => ({
            role: msg.role,
            content: msg.content,
          })),
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get response");
      }

      const data = await response.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.message || "Sorry, I couldn't process your request.",
        role: "assistant",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
      return data.message || "Sorry, I couldn't process your request.";
    } catch (error) {
      console.error("Error sending message:", error);

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content:
          "Sorry, there was an error processing your request. Please try again.",
        role: "assistant",
        timestamp: new Date(),
      };

      return Promise.reject("Error sending message.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(inputMessage);
    }
  };

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
      <p>{designResult}</p>
    </div>
  );

  return (
    <div className="w-full">
      <DropDown
        text="Cleaning Data"
        view={CleaningStepData}
      />
      {currentStep > 0 && (
        <DropDown
          text="Designing Analysis Procedure"
          view={DesignStepData}
        />
      )}
      {currentStep > 1 && (
        <DropDown
          text="Running Statistical Tests"
          view={{
            images: hypothesisTestingResult.figures || [],
            p_vals: hypothesisTestingResult.p_values || [],
          }}
        />
      )}
      {currentStep > 2 && (
        <DropDown
          text="Found Data!"
          view={tempData}
        />
      )}
      <div className="mt-8">
        <Chat
          initialMessages={[
            {
              id: "1",
              content: "Hello! I'm your data analysis assistant. Ask me anything about your analysis.",
              role: "assistant",
              timestamp: new Date(),
            },
          ]}
          onSendMessage={sendMessage}
          title="Data Analysis Assistant"
        />
      </div>
    </div>
  );
}
