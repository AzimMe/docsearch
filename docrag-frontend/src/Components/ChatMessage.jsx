export default function ChatMessage({ message, type }) {
  if (type === 'user') {
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-xs lg:max-w-md px-4 py-2 bg-blue-600 text-white rounded-lg">
          <p className="text-sm">{message}</p>
        </div>
      </div>
    );
  } else {
    return (
      <div className="flex justify-start mb-4">
        <div className="max-w-xs lg:max-w-md px-4 py-2 bg-gray-700 text-white rounded-lg">
          <p className="text-sm">{message}</p>
        </div>
      </div>
    );
  }
}