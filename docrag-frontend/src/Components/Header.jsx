import FileUpload from './FileUpload';

export default function Header() {
  return (
    <header className="flex justify-between items-center p-4 bg-gray-800 border-b border-gray-700">
      <h1 className="text-xl font-bold text-white">Document Chat</h1>
      <FileUpload />
    </header>
  );
}