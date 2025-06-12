import { useState } from 'react';
import Header from './Components/Header';
import ChatMessage from './Components/ChatMessage';
import ChatInput from './Components/ChatInput';
import Sidebar from './Components/Sidebar';

function App() {
  const [messages, setMessages] = useState([]);
  const [filepaths, setFilepaths] = useState([]);

  const sendQuestion = async (question) => {
    setMessages((prev) => [...prev, { type: 'user', message: question }]);

    try {
      const res = await fetch('http://localhost:3001/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      });

      const data = await res.json();
      setMessages((prev) => [...prev, { type: 'ai', message: data.answer }]);

      if (Array.isArray(data.filepaths)) {
        setFilepaths(data.filepaths);
      }
    } catch (err) {
      setMessages((prev) => [...prev, { type: 'ai', message: '⚠️ Error contacting the server.' }]);
    }
  };

  const handleAsk = (question) => {
    if (question.trim()) {
      sendQuestion(question);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white">
      {/* Header */}
      <Header />

      <div className="flex flex-1 overflow-hidden">
        {/* Chat Area */}
        <main className="flex flex-col flex-1">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-1">
            {messages.length === 0 ? (
              <div className="flex items-center justify-center h-full text-gray-400">
                <p>Start by uploading documents and asking questions!</p>
              </div>
            ) : (
              messages.map((msg, idx) => (
                <ChatMessage key={idx} message={msg.message} type={msg.type} />
              ))
            )}
          </div>

          {/* Input Area */}
          <ChatInput onSendMessage={handleAsk} />
        </main>

        {/* Sidebar */}
        <Sidebar filepaths={filepaths} />
      </div>
    </div>
  );
}

export default App;