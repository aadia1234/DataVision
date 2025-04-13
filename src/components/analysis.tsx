"use client";

import { useState, useEffect, useRef } from "react";
import DropDown from "@/components/drop-down";
import { Button } from "@/components/ui/button";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { Input } from "@/components/ui/input";
import { Send, Paperclip } from "lucide-react";

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

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;

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
          message: inputMessage,
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
    } catch (error) {
      console.error("Error sending message:", error);

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content:
          "Sorry, there was an error processing your request. Please try again.",
        role: "assistant",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
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
        clicked={currentStep}
        phaseNum={0}
        data={CleaningStepData}
      />
      {currentStep > 0 && (
        <DropDown
          text="Designing Analysis Procedure"
          clicked={currentStep}
          phaseNum={1}
          data={DesignStepData}
        />
      )}
      {currentStep > 1 && (
        <ImageDropDown
          text="Running Statistical Tests"
          clicked={currentStep}
          phaseNum={2}
          images={hypothesisTestingResult.figures || []}
          p_vals={hypothesisTestingResult.p_values || []}
        />
      )}
      {currentStep > 2 && (
        <DropDown
          text="Found Data!"
          clicked={currentStep}
          phaseNum={3}
          data={tempData}
        />
      )}

      <div className="mt-8 border rounded-lg shadow-md">
        <div className="bg-gray-100 p-4 rounded-t-lg border-b">
          <h3 className="font-semibold text-gray-800">
            Data Analysis Assistant
          </h3>
        </div>

        <div className="h-80 overflow-y-auto p-4 space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${
                message.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`max-w-3/4 p-3 rounded-lg ${
                  message.role === "user"
                    ? "bg-blue-500 text-white rounded-br-none"
                    : "bg-gray-200 text-gray-800 rounded-bl-none"
                }`}
              >
                {message.content}
                <div
                  className={`text-xs mt-1 ${
                    message.role === "user" ? "text-blue-100" : "text-gray-500"
                  }`}
                >
                  {message.timestamp.toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </div>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-gray-200 text-gray-800 p-3 rounded-lg rounded-bl-none">
                <div className="flex space-x-2">
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                  <div
                    className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
                    style={{ animationDelay: "0.2s" }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
                    style={{ animationDelay: "0.4s" }}
                  ></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="p-3 border-t flex gap-2">
          <Button variant="outline" size="icon" className="shrink-0">
            <Paperclip className="h-5 w-5" />
          </Button>
          <Input
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Type your message..."
            className="flex-1"
          />
          <Button
            onClick={sendMessage}
            disabled={isLoading || !inputMessage.trim()}
            className="shrink-0"
          >
            <Send className="h-5 w-5 mr-1" />
            Send
          </Button>
        </div>
      </div>
    </div>
  );
}
