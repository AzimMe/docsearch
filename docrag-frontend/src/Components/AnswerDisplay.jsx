export default function AnswerDisplay({ answer }) {
  return (
    <div className="p-4 border rounded shadow bg-gray-50">
      <h2 className="text-xl font-semibold mb-2">💡 Answer</h2>
      <p>{answer || 'No answer yet.'}</p>
    </div>
  );
}
