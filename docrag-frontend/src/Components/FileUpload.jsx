import { useState } from 'react';

export default function FileUpload() {
  const [files, setFiles] = useState(null);
  const [message, setMessage] = useState('');

  const handleUpload = async () => {
    if (!files || files.length === 0) return;
    
    const formData = new FormData();
    for (let file of files) {
      formData.append('documents', file);
    }

    try {
      const res = await fetch('http://localhost:3001/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      setMessage(data.message);
    } catch (err) {
      setMessage('Upload failed');
    }
  };

  return (
    <div className="flex items-center gap-4">
      <div className="flex items-center gap-2">
        <input
          type="file"
          multiple
          onChange={(e) => setFiles(e.target.files)}
          className="text-sm text-gray-300 file:mr-2 file:py-1 file:px-3 file:rounded-md file:border-0 file:bg-blue-600 file:text-white file:cursor-pointer hover:file:bg-blue-700"
        />
        <button 
          onClick={handleUpload}
          className="bg-blue-600 hover:bg-blue-700 px-3 py-1 rounded text-white text-sm font-medium transition-colors"
        >
          Upload
        </button>
      </div>
      {message && (
        <span className="text-sm text-green-400">{message}</span>
      )}
    </div>
  );
}