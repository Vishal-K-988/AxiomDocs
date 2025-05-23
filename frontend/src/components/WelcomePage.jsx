import React, { useState, useEffect } from 'react';
import { UserButton, useAuth } from '@clerk/clerk-react';
import { Plus, Send } from 'lucide-react';
import FileUploadModal from './FileUploadModal';
import Sidebar from './Sidebar';
import { fileService } from '../services/fileService';

const WelcomePage = () => {
  const { isLoaded, isSignedIn } = useAuth();
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [question, setQuestion] = useState('');
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch files on component mount and when auth state changes
  useEffect(() => {
    if (isLoaded && isSignedIn) {
      fetchFiles();
    }
  }, [isLoaded, isSignedIn]);

  const fetchFiles = async () => {
    try {
      setIsLoading(true);
      const files = await fileService.getFiles();
      setUploadedFiles(files);
      setError(null);
    } catch (err) {
      if (err.message === 'Not authenticated') {
        setError('Please sign in to access your files.');
      } else {
        setError('Failed to load files. Please try again later.');
      }
      console.error('Error fetching files:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (files) => {
    try {
      // The files are already uploaded by the FileUploadModal
      // We just need to update our local state
      setUploadedFiles(prev => [...prev, ...files]);
      setError(null);
    } catch (err) {
      setError('Failed to update file list. Please refresh the page.');
      console.error('Error updating file list:', err);
    }
  };

  const handleRenameFile = async (fileId, newName) => {
    try {
      await fileService.renameFile(fileId, newName);
      setUploadedFiles(prev => 
        prev.map(file => 
          file.id === fileId 
            ? { ...file, name: newName }
            : file
        )
      );
      setError(null);
    } catch (err) {
      setError('Failed to rename file. Please try again.');
      console.error('Error renaming file:', err);
    }
  };

  const handleDeleteFile = async (fileId) => {
    try {
      await fileService.deleteFile(fileId);
      setUploadedFiles(prev => prev.filter(file => file.id !== fileId));
      setError(null);
    } catch (err) {
      setError('Failed to delete file. Please try again.');
      console.error('Error deleting file:', err);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!question.trim()) return;
    // Handle question submission
    setQuestion('');
  };

  const toggleSidebar = () => {
    setIsSidebarCollapsed(prev => !prev);
  };

  if (!isLoaded) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-green-600"></div>
      </div>
    );
  }

  if (!isSignedIn) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Please Sign In</h2>
          <p className="text-gray-600 mb-4">You need to be signed in to access your files.</p>
          <UserButton />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="w-full border-b border-gray-200 bg-white shadow-sm">
        <div className="container mx-auto px-6 py-4 flex items-center justify-between">
          {/* Logo or App Name */}
          <div className="flex items-center gap-2">
            <img src="/src/assests/logo.png" alt="Logo" className="h-8 w-8" />
            <span className="font-bold text-xl text-gray-900">ai <span className="text-green-600">planet</span></span>
            <span className="text-xs text-gray-500 ml-2">formerly DPhi</span>
          </div>
          {/* Right side: Upload PDF & UserButton */}
          <div className="flex items-center gap-4">
            <button 
              onClick={() => setIsUploadModalOpen(true)}
              className="flex items-center gap-2 bg-green-600 text-white rounded-md px-4 py-2 hover:bg-green-700 transition-colors font-medium shadow-sm"
            >
              <Plus className="w-4 h-4" /> Upload Files
            </button>
            <div className="size-7">
              <UserButton
                appearance={{
                  elements: {
                    userButtonBox: "cl-userButtonBox",
                    userButtonTrigger:
                      "cl-userButtonTrigger cl-button after:absolute after:-inset-1 after:rounded-full after:border-2 after:border-blue-900 after:opacity-0 focus-visible:after:opacity-100 focus:shadow-none",
                    userButtonAvatarBox: "cl-avatarBox cl-userButtonAvatarBox",
                    userButtonAvatarImage: "cl-avatarImage cl-userButtonAvatarImage",
                    userButtonPopoverCard: "shadow-lg border border-gray-100",
                    userPreview: "border-b border-gray-100",
                    userButtonPopoverActions: "p-2",
                    userButtonPopoverActionButton: "hover:bg-gray-100 rounded-md",
                    userButtonPopoverFooter: "border-t border-gray-100",
                  },
                }}
              />
            </div>
          </div>
        </div>
      </header>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border-l-4 border-red-400 p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Main Content with Sidebar */}
      <main className="flex-1 flex">
        <div className="flex-1 p-6">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-green-600"></div>
            </div>
          ) : (
            <div className="text-center text-gray-500">
              Select a file from the sidebar to view its contents
            </div>
          )}
        </div>
        <Sidebar 
          files={uploadedFiles} 
          isCollapsed={isSidebarCollapsed}
          onToggle={toggleSidebar}
          onRename={handleRenameFile}
          onDelete={handleDeleteFile}
        />
      </main>

      {/* Bottom Prompt Input */}
      <footer className="w-full border-t border-gray-200 bg-white fixed bottom-0 left-0 z-20">
        <div className="container mx-auto px-4 py-4">
          <form onSubmit={handleSubmit} className="w-full flex items-center gap-2">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question about your documents..."
              className="flex-1 rounded-lg border border-gray-200 px-4 py-3 bg-gray-50 text-gray-900 focus:outline-none focus:ring-2 focus:ring-green-200 focus:border-transparent"
            />
            <button
              type="submit"
              disabled={!question.trim()}
              className="bg-green-600 hover:bg-green-700 text-white px-4 py-3 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              <Send className="w-5 h-5" />
            </button>
          </form>
        </div>
      </footer>

      {/* File Upload Modal */}
      <FileUploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onUpload={handleFileUpload}
      />
    </div>
  );
};

export default WelcomePage; 