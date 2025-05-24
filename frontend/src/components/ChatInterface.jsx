import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import logo from '../assests/logo.png';

const ChatInterface = ({ selectedFile, chatHistory = [], onSendMessage, isLoading, userProfileImage }) => {
  const [inputMessage, setInputMessage] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory, isLoading]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;
    await onSendMessage(inputMessage);
    setInputMessage('');
  };

  return (
    <div className="flex flex-col h-full bg-white rounded-lg shadow-sm border border-gray-200">
      {/* Chat Header */}
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-900">
          {selectedFile ? `Chat about ${selectedFile.name}` : 'Select a file to start chatting'}
        </h2>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {chatHistory.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <img src={logo} alt="Bot" className="w-12 h-12 mb-4 rounded-full object-contain bg-white p-1 border border-green-200" />
            <p className="text-lg font-medium">Start a conversation</p>
            <p className="text-sm">Ask questions about your document</p>
          </div>
        ) : (
          chatHistory.map((message, idx) => (
            <div
              key={idx}
              className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`flex items-start gap-2 max-w-[80%] ${
                  message.sender === 'user'
                    ? 'flex-row-reverse'
                    : 'flex-row'
                }`}
              >
                <div
                  className={`w-9 h-9 rounded-full flex items-center justify-center overflow-hidden border border-green-200 bg-white ${
                    message.sender === 'user' ? 'bg-green-100 text-green-600 border-none' : ''
                  }`}
                  style={{ minWidth: 36, minHeight: 36 }}
                >
                  {message.sender === 'user' ? (
                    userProfileImage ? (
                      <img
                        src={userProfileImage}
                        alt="User"
                        className="w-9 h-9 rounded-full object-cover"
                      />
                    ) : (
                      <User className="w-5 h-5" />
                    )
                  ) : (
                    <img
                      src={logo}
                      alt="Bot"
                      className="w-8 h-8 object-contain p-1"
                      style={{ background: 'white', borderRadius: '50%' }}
                      onError={e => {
                        e.target.style.display = 'none';
                        e.target.parentNode.innerHTML = `<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' class='lucide lucide-bot-icon lucide-bot'><path d='M12 8V4H8'/><rect width='16' height='12' x='4' y='8' rx='2'/><path d='M2 14h2'/><path d='M20 14h2'/><path d='M15 13v2'/><path d='M9 13v2'/></svg>`;
                      }}
                    />
                  )}
                </div>
                <div
                  className={`rounded-lg px-4 py-2 ml-1 ${
                    message.sender === 'user'
                      ? 'bg-green-600 text-white'
                      : 'bg-gray-50 text-gray-900 border border-green-100 shadow-sm'
                  }`}
                  style={message.sender === 'ai' ? { minWidth: 0, wordBreak: 'break-word', background: '#f8fefb', marginLeft: 8, maxWidth: '100%' } : {}}
                >
                  {message.sender === 'ai' ? (
                    <ReactMarkdown>{message.content}</ReactMarkdown>
                  ) : (
                    <p className="text-sm">{message.content}</p>
                  )}
                  <span className="text-xs opacity-70 mt-1 block">
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </span>
                </div>
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="flex justify-start">
            <div className="flex items-start gap-2">
              <div className="w-8 h-8 rounded-full bg-white border border-green-200 flex items-center justify-center overflow-hidden">
                <img src={logo} alt="Bot" className="w-8 h-8 rounded-full object-contain bg-white p-1 border border-green-200" />
              </div>
              <div className="bg-gray-100 rounded-lg px-4 py-2">
                <div className="flex space-x-2">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100" />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200" />
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <form onSubmit={handleSendMessage} className="p-4 border-t border-gray-200">
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 rounded-lg border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
            disabled={!selectedFile || isLoading}
          />
          <button
            type="submit"
            disabled={!inputMessage.trim() || !selectedFile || isLoading}
            className="bg-green-600 text-white p-2 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatInterface; 