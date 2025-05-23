import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor to include auth token and user ID
api.interceptors.request.use(async (config) => {
  // Get the session token from Clerk
  const token = await window.Clerk?.session?.getToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }

  // Get the user ID from Clerk
  const userId = window.Clerk?.user?.id;
  if (userId) {
    config.headers['X-User-ID'] = userId;
  }

  return config;
});

export const fileService = {
  // Upload file with progress tracking
  uploadFile: async (file, onProgress) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await api.post('/files/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          onProgress?.(percentCompleted);
        },
      });
      return response.data;
    } catch (error) {
      throw handleApiError(error);
    }
  },

  // Get all files
  getFiles: async () => {
    try {
      const response = await api.get('/files');
      return response.data;
    } catch (error) {
      throw handleApiError(error);
    }
  },

  // Delete file
  deleteFile: async (fileId) => {
    try {
      await api.delete(`/files/${fileId}`);
      return true;
    } catch (error) {
      throw handleApiError(error);
    }
  },

  // Rename file
  renameFile: async (fileId, newName) => {
    try {
      const response = await api.patch(`/files/${fileId}/rename`, {
        new_filename: newName,
      });
      return response.data;
    } catch (error) {
      throw handleApiError(error);
    }
  },
};

// Error handling helper
const handleApiError = (error) => {
  if (error.response) {
    // The request was made and the server responded with a status code
    // that falls out of the range of 2xx
    const message = error.response.data?.detail || error.response.data?.message || 'An error occurred';
    if (typeof message === 'object') {
      // Handle validation errors
      const errorDetails = Array.isArray(message) ? message : [message];
      const errorMessage = errorDetails
        .map(err => `${err.loc?.join('.')}: ${err.msg}`)
        .join(', ');
      throw new Error(errorMessage);
    }
    throw new Error(message);
  } else if (error.request) {
    // The request was made but no response was received
    throw new Error('No response from server. Please check your connection.');
  } else {
    // Something happened in setting up the request that triggered an Error
    throw new Error('Error setting up the request.');
  }
}; 