export default function Sidebar({ filepaths }) {
  return (
    <aside className="w-80 bg-gray-800 border-l border-gray-700 p-4 overflow-y-auto">
      <div className="mb-6">
        <h2 className="font-bold text-lg mb-3 flex items-center gap-2 text-white">
          📁 Relevant Files
        </h2>
        {filepaths.length === 0 ? (
          <p className="text-gray-400 text-sm">No files referenced yet</p>
        ) : (
          <ul className="space-y-2">
            {filepaths.map((path, idx) => (
              <li key={idx} className="text-sm text-gray-300 bg-gray-700 p-2 rounded">
                {path.split('/').pop()}
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  );
}