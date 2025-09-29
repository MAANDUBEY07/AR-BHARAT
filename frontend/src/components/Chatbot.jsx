import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';

const API_BASE = process.env.NODE_ENV === 'production' ? '' : 'http://localhost:5000';

export default function Chatbot({ patternMetadata = null, isOpen, onClose }) {
  const [messages, setMessages] = useState([
    {
      type: 'bot',
      content: '🙏 Namaste! I\'m your AR BHARAT cultural assistant. Ask me about Kolam patterns, Indian art traditions, or how to create beautiful AR experiences! 🎨',
      timestamp: new Date().toISOString()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [aiEnabled, setAiEnabled] = useState(true);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      type: 'user',
      content: inputMessage,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE}/api/chatbot/chat`, {
        message: inputMessage,
        metadata: patternMetadata || {},
        history: messages.slice(-6), // Last 6 messages for context
        use_ai: aiEnabled
      });

      const botMessage = {
        type: 'bot',
        content: response.data.response,
        timestamp: response.data.timestamp,
        aiPowered: response.data.ai_powered
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage = {
        type: 'bot',
        content: '🤖 Sorry, I encountered an issue. Please try again or check if your OpenAI API key is configured correctly.',
        timestamp: new Date().toISOString(),
        error: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const explainCurrentPattern = async () => {
    if (!patternMetadata || isLoading) return;

    setIsLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/api/chatbot/explain`, {
        metadata: patternMetadata,
        use_ai: aiEnabled
      });

      const botMessage = {
        type: 'bot',
        content: response.data.explanation,
        timestamp: new Date().toISOString(),
        aiPowered: response.data.ai_powered
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Explanation error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const quickQuestions = [
    'Tell me about Kolam traditions',
    'How do I create better patterns?',
    'What is the cultural significance?',
    'How does AR enhance the experience?'
  ];

  const handleQuickQuestion = (question) => {
    setInputMessage(question);
  };

  if (!isOpen) return null;

  return (
    <motion.div
      initial={{ opacity: 0, x: 300 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 300 }}
      className="fixed right-4 bottom-4 w-96 h-[500px] bg-white rounded-xl shadow-2xl border border-gray-200 flex flex-col z-50"
    >
      {/* Header */}
      <div className="bg-gradient-to-r from-orange-500 via-green-500 to-blue-500 text-white p-4 rounded-t-xl flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center">
            🤖
          </div>
          <div>
            <h3 className="font-semibold">AR BHARAT Assistant</h3>
            <p className="text-xs opacity-90">
              {aiEnabled ? '✨ AI-Powered' : '🔧 Rule-based'}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setAiEnabled(!aiEnabled)}
            className="text-xs px-2 py-1 bg-white/20 rounded hover:bg-white/30 transition-colors"
            title="Toggle AI mode"
          >
            {aiEnabled ? 'AI' : 'BASIC'}
          </button>
          <button
            onClick={onClose}
            className="text-white hover:text-gray-200 transition-colors"
          >
            ✕
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] p-3 rounded-lg ${
                message.type === 'user'
                  ? 'bg-blue-500 text-white rounded-br-sm'
                  : message.error
                  ? 'bg-red-50 text-red-800 border border-red-200 rounded-bl-sm'
                  : 'bg-gray-100 text-gray-800 rounded-bl-sm'
              }`}
            >
              <p className="text-sm whitespace-pre-wrap">{message.content}</p>
              {message.aiPowered && (
                <div className="text-xs opacity-70 mt-1">✨ AI-powered</div>
              )}
            </div>
          </motion.div>
        ))}
        
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex justify-start"
          >
            <div className="bg-gray-100 p-3 rounded-lg rounded-bl-sm">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
              </div>
            </div>
          </motion.div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Quick Actions */}
      {patternMetadata && (
        <div className="px-4 py-2 border-t border-gray-100">
          <button
            onClick={explainCurrentPattern}
            disabled={isLoading}
            className="w-full text-sm bg-gradient-to-r from-orange-500/10 via-green-500/10 to-blue-500/10 text-gray-700 px-3 py-2 rounded-lg hover:from-orange-500/20 hover:via-green-500/20 hover:to-blue-500/20 transition-all disabled:opacity-50"
          >
            🎨 Explain Current Pattern
          </button>
        </div>
      )}

      {/* Quick Questions */}
      <div className="px-4 py-2 border-t border-gray-100">
        <div className="flex flex-wrap gap-1">
          {quickQuestions.map((question, index) => (
            <button
              key={index}
              onClick={() => handleQuickQuestion(question)}
              className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded-full hover:bg-gray-200 transition-colors"
            >
              {question}
            </button>
          ))}
        </div>
      </div>

      {/* Input */}
      <div className="p-4 border-t border-gray-100">
        <div className="flex gap-2">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Ask about Kolam, Indian art, or AR experiences..."
            className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
            disabled={isLoading}
          />
          <button
            onClick={sendMessage}
            disabled={isLoading || !inputMessage.trim()}
            className="px-4 py-2 bg-gradient-to-r from-orange-500 via-green-500 to-blue-500 text-white rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 text-sm"
          >
            Send
          </button>
        </div>
      </div>
    </motion.div>
  );
}