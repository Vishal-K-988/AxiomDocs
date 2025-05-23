import React, { useState } from 'react';
import { FileText, Clock, FileIcon, Image, File, ChevronRight, MoreVertical, Pencil, Trash2, X, Check } from 'lucide-react';

const getFileIcon = (fileName) => {
  if (!fileName) return <FileText className="w-5 h-5 text-gray-600" />;
  
  const extension = fileName.split('.').pop().toLowerCase();
  switch (extension) {
    case 'pdf':
      return <File className="w-5 h-5 text-red-600" />;
    case 'jpg':
    case 'jpeg':
    case 'png':
    case 'gif':
      return <Image className="w-5 h-5 text-blue-600" />;
    default:
      return <FileText className="w-5 h-5 text-green-600" />;
  }
};

const formatFileSize = (bytes) => {
  if (!bytes) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

const FileCard = ({ file, onRename, onDelete }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isRenaming, setIsRenaming] = useState(false);
  const [newName, setNewName] = useState(file?.name || '');

  const handleRename = () => {
    if (newName.trim() && newName !== file?.name) {
      onRename(file.id, newName);
    }
    setIsRenaming(false);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleRename();
    } else if (e.key === 'Escape') {
      setIsRenaming(false);
      setNewName(file?.name || '');
    }
  };

  if (!file) return null;

  return (
    <div className="group p-3 rounded-lg border border-gray-200 hover:border-green-500 hover:shadow-md transition-all duration-200 cursor-pointer bg-white">
      <div className="flex items-start gap-3">
        <div className="p-2 bg-gray-50 rounded-md group-hover:bg-green-50 transition-colors">
          {getFileIcon(file.name)}
        </div>
        <div className="flex-1 min-w-0">
          {isRenaming ? (
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                onKeyDown={handleKeyPress}
                className="flex-1 text-sm font-medium text-gray-900 border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-green-500"
                autoFocus
              />
              <button
                onClick={handleRename}
                className="p-1 text-green-600 hover:bg-green-50 rounded"
              >
                <Check className="w-4 h-4" />
              </button>
              <button
                onClick={() => {
                  setIsRenaming(false);
                  setNewName(file.name || '');
                }}
                className="p-1 text-gray-500 hover:bg-gray-50 rounded"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <p className="text-sm font-medium text-gray-900 truncate group-hover:text-green-600 transition-colors">
              {file.name || 'Unnamed File'}
            </p>
          )}
          <div className="flex items-center gap-3 mt-1">
            <div className="flex items-center gap-1">
              <Clock className="w-3 h-3 text-gray-400" />
              <span className="text-xs text-gray-500">
                {file.uploadTime ? new Date(file.uploadTime).toLocaleString() : 'Unknown date'}
              </span>
            </div>
            {file.size && (
              <span className="text-xs text-gray-500">
                {formatFileSize(file.size)}
              </span>
            )}
          </div>
        </div>
        <div className="relative">
          <button
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            className="p-1 text-gray-400 hover:text-gray-600 rounded-full hover:bg-gray-100 transition-colors"
          >
            <MoreVertical className="w-4 h-4" />
          </button>
          
          {isMenuOpen && (
            <div className="absolute right-0 mt-1 w-48 bg-white rounded-md shadow-lg border border-gray-200 py-1 z-10">
              <button
                onClick={() => {
                  setIsRenaming(true);
                  setIsMenuOpen(false);
                }}
                className="w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 flex items-center gap-2"
              >
                <Pencil className="w-4 h-4" />
                Rename
              </button>
              <button
                onClick={() => {
                  onDelete(file.id);
                  setIsMenuOpen(false);
                }}
                className="w-full px-4 py-2 text-sm text-red-600 hover:bg-red-50 flex items-center gap-2"
              >
                <Trash2 className="w-4 h-4" />
                Delete
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const Sidebar = ({ files = [], isCollapsed, onToggle, onRename, onDelete }) => {
  return (
    <div className={`relative flex transition-all duration-300 ease-in-out ${isCollapsed ? 'w-0' : 'w-80'}`}>
      <div className={`border-l border-gray-200 bg-white h-full overflow-y-auto transition-all duration-300 ${isCollapsed ? 'w-0 opacity-0' : 'w-80 opacity-100'}`}>
        <div className="p-4">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold text-gray-900">Documents</h2>
            <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded-full">
              {files.length} files
            </span>
          </div>
          
          <div className="space-y-3">
            {files.map((file) => (
              <FileCard
                key={file.id}
                file={file}
                onRename={onRename}
                onDelete={onDelete}
              />
            ))}
            
            {files.length === 0 && (
              <div className="text-center py-12">
                <div className="w-16 h-16 mx-auto mb-4 bg-gray-50 rounded-full flex items-center justify-center">
                  <FileIcon className="w-8 h-8 text-gray-400" />
                </div>
                <p className="text-gray-900 font-medium mb-1">No documents yet</p>
                <p className="text-sm text-gray-500">
                  Upload files to see them here
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Toggle Button */}
      <button
        onClick={onToggle}
        className="absolute -left-4 top-1/2 -translate-y-1/2 bg-white border border-gray-200 rounded-full p-1.5 shadow-md hover:bg-gray-50 transition-colors focus:outline-none focus:ring-2 focus:ring-green-500"
        aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
      >
        <ChevronRight className={`w-4 h-4 text-gray-600 transition-transform duration-300 ${isCollapsed ? 'rotate-180' : ''}`} />
      </button>
    </div>
  );
};

export default Sidebar; 