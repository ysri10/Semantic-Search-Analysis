import React, { useState } from 'react';
import axios from 'axios';

function FileUpload({ onUpload, isUploading, setIsUploading }) {
    const [files, setFiles] = useState(null);
    const [error, setError] = useState('');
    const [successMessage, setSuccessMessage] = useState('');

    const handleFileChange = (event) => {
        setFiles(event.target.files);
        setSuccessMessage('');
        setError('');
    };

    const handleSubmit = async () => {
        if (!files || files.length === 0) {
            setError('Please select at least one file.');
            return;
        }

        const formData = new FormData();
        Array.from(files).forEach((file) => {
            formData.append('files', file);
        });

        setIsUploading(true);

        try {
            const response = await axios.post('http://localhost:5001/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                }
            });

            console.log(response.data);
            setSuccessMessage('Files uploaded successfully!');
            onUpload(response.data.file_map);
        } catch (error) {
            console.error("Error uploading files:", error);
            setError("Error uploading files. Please try again.");
        } finally {
            setIsUploading(false);
        }
    };

    return (
        <div className="mb-4">
            <h2 className="text-xl font-semibold mb-2">Upload Your Files</h2>
            <input
                type="file"
                multiple
                onChange={handleFileChange}
                className="mb-2 p-2 border rounded"
            />
            <button
                onClick={handleSubmit}
                disabled={isUploading}
                className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:bg-blue-300"
            >
                {isUploading ? 'Uploading...' : 'Upload Files'}
            </button>
            {successMessage && <p className="mt-2 text-green-500">{successMessage}</p>}
            {error && <p className="mt-2 text-red-500">{error}</p>}
        </div>
    );
}

export default FileUpload;