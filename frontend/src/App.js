import React, { useState } from 'react';
import axios from 'axios';

const App = () => {
  const [files, setFiles] = useState([]);
  const [uploadStatus, setUploadStatus] = useState('');
  const [fileMap, setFileMap] = useState({});
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [error, setError] = useState('');

  // Handle multiple file selection
  const handleFileChange = (event) => {
    setFiles(event.target.files);
  };

  // Handle file upload
  const handleUpload = async () => {
    setUploadStatus('Uploading...');
    setError('');

    const formData = new FormData();
    // Append each file to formData
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    try {
      const response = await axios.post('http://localhost:5001/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setUploadStatus('Files uploaded successfully');
      setFileMap(response.data.file_map);  // Update the fileMap state with response
    } catch (error) {
      setUploadStatus('Upload failed');
      setError(error.response?.data?.error || error.message);
      console.error('Error:', error);
    }
  };

  // Handle search
  const handleSearch = async () => {
    setError('');
    try {
      const response = await axios.post('http://localhost:5001/search', { query: searchQuery });
      setSearchResults(response.data.results);
    } catch (error) {
      setError(error.response?.data?.error || error.message);
      console.error('Error:', error);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-4">Semantic Search Application</h1>
      
      {/* File Upload Section */}
      <div className="mb-4">
        <h2 className="text-xl font-semibold mb-2">Upload Your Files</h2>
        <input
          type="file"
          onChange={handleFileChange}
          multiple
          className="mb-2 p-2 border rounded"
        />
        <button
          onClick={handleUpload}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          Upload Files
        </button>
        <p className="mt-2">{uploadStatus}</p>
      </div>

      {/* Search Section */}
      <div className="mb-4">
        <h2 className="text-xl font-semibold mb-2">Search</h2>
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Enter your search query"
          className="mr-2 p-2 border rounded"
        />
        <button
          onClick={handleSearch}
          className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
        >
          Search
        </button>
      </div>

      {/* Display Error */}
      {error && <p className="text-red-500 mb-4">{error}</p>}

      {/* Search Results Section */}
      <div>
        <h2 className="text-xl font-semibold mb-2">Search Results</h2>
        {searchResults.length > 0 ? (
          searchResults.map((result, index) => (
            <div key={index} className="mb-4 p-4 border rounded">
              <h3 className="font-semibold">{result.file_name}</h3>
              <p>{result.content}</p>
            </div>
          ))
        ) : (
          <p>No results found. Try uploading some files and searching again.</p>
        )}
      </div>

      {/* Uploaded Files Section */}
      <div className="mt-4">
        <h2 className="text-xl font-semibold mb-2">Uploaded Files</h2>
        {Object.keys(fileMap).length > 0 ? (
          <ul>
            {Object.keys(fileMap).map((fileName) => (
              <li key={fileName}>{fileName}</li>
            ))}
          </ul>
        ) : (
          <p>No files uploaded yet.</p>
        )}
      </div>
    </div>
  );
};

export default App;
