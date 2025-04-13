'use client';

import { useState } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: "Hi! I'm a chat assistant. How can I help you today?",
    },
  ]);
  const [message, setMessage] = useState<string>('');

  const sendMessage = async () => {
    if (!message.trim()) return;

    const userMessage: Message = { role: 'user', content: message };

    setMessages((prevMessages) => [
      ...prevMessages,
      userMessage,
      { role: 'assistant', content: '' },
    ]);
    setMessage('');

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify([...messages, userMessage]),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const text = decoder.decode(value, { stream: true });

          setMessages((prevMessages) => {
            const lastMessage = prevMessages[prevMessages.length - 1];
            const otherMessages = prevMessages.slice(0, prevMessages.length - 1);

            return [
              ...otherMessages,
              { ...lastMessage, content: lastMessage.content + text },
            ];
          });
        }
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          role: 'assistant',
          content: 'Sorry, there was an error processing your request.',
        },
      ]);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center w-screen h-screen bg-gray-100">
      <div className="flex flex-col w-full max-w-lg h-[700px] border border-gray-300 bg-white rounded-lg shadow-lg p-4">
        <div className="flex flex-col flex-grow overflow-y-auto space-y-4">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${
                message.role === 'assistant' ? 'justify-start' : 'justify-end'
              }`}
            >
              <div
                className={`${
                  message.role === 'assistant'
                    ? 'bg-blue-500 text-white'
                    : 'bg-green-500 text-white'
                } rounded-lg p-3 max-w-xs`}
              >
                {message.content}
              </div>
            </div>
          ))}
        </div>

        <div className="flex items-center space-x-2 mt-4">
          <input
            type="text"
            className="flex-grow border border-gray-300 rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Type your message..."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
          />
          <button
            onClick={sendMessage}
            className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chat;