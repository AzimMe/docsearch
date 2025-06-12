import { useState } from 'react';

export default function QuestionBox({ onAsk }) {
  const [question, setQuestion] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (question.trim()) {
      onAsk(question);
      setQuestion('');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="mb-6 p-4 border rounded shadow">
      <h2 className="text-xl font-semibold mb-2">❔ Ask a Question</h2>
      <input
        type="text"
        placeholder="Type your question..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        className="border p-2 rounded w-full mb-2"
      />
      <button className="bg-green-500 text-white px-4 py-1 rounded">Ask</button>
    </form>
  );
}
