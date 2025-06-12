export default function ChatInput({ onSendMessage }) {
  const handleSend = (input) => {
    const question = input.value;
    if (question.trim()) {
      onSendMessage(question);
      input.value = '';
    }
  };

  return (
    <div className="p-4 bg-gray-800 border-t border-gray-700">
      <div className="flex gap-2 max-w-4xl mx-auto">
        <input
          ref={(input) => {
            if (input) {
              input.onKeyDown = (e) => {
                if (e.key === 'Enter') {
                  handleSend(input);
                }
              };
            }
          }}
          placeholder="Ask a question about your documents..."
          className="flex-1 p-3 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none placeholder-gray-400"
        />
        <button 
          onClick={(e) => {
            const input = e.target.parentElement.querySelector('input');
            handleSend(input);
          }}
          className="bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded-lg text-white font-medium transition-colors"
        >
          Send
        </button>
      </div>
    </div>
  );
}