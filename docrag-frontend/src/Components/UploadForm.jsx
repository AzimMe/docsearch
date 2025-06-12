import { useState } from 'react';

export default function UploadForm() {
  const [files, setFiles] = useState(null);
  const [message, setMessage] = useState('');

  const handleUpload = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    for (let file of files) {
      formData.append('documents', file);
    }

    const res = await fetch('http://localhost:3001/upload', {
      method: 'POST',
      body: formData
    });

    const data = await res.json();
    setMessage(data.message);
  };

  return (
    <div className="p-4 border rounded max-w-md mx-auto mt-4">
      <h2 className="text-xl font-semibold mb-2">Upload Documents</h2>
      <form onSubmit={handleUpload}>
        <input
          type="file"
          multiple
          onChange={(e) => setFiles(e.target.files)}
          className="mb-2"
        />
        <button className="bg-blue-500 text-white px-4 py-1 rounded">Upload</button>
      </form>
      {message && <p className="mt-2 text-green-600">{message}</p>}
    </div>
  );
}
