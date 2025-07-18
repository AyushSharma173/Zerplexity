// frontend/src/App.jsx

import { useState, useRef, useEffect } from 'react';
import { askStream } from './api';
import RenderAnswer from './utils/renderAnswer.jsx';

// Chat storage utilities
const CHAT_STORAGE_KEY = 'better-perplexity-chats';

function loadChatsFromStorage() {
  try {
    const stored = localStorage.getItem(CHAT_STORAGE_KEY);
    return stored ? JSON.parse(stored) : {};
  } catch (error) {
    console.error('Failed to load chats from storage:', error);
    return {};
  }
}

function saveChatsToStorage(chats) {
  try {
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(chats));
  } catch (error) {
    console.error('Failed to save chats to storage:', error);
  }
}

export default function App() {
  const [chats, setChats] = useState(loadChatsFromStorage);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState('vanilla');
  const [model, setModel] = useState('gpt4o');
  const [training, setTraining] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [showSourceSelection, setShowSourceSelection] = useState(false);
  const [selectedSources, setSelectedSources] = useState([]);
  const [currentTrainingId, setCurrentTrainingId] = useState(null);
  const timelineRef = useRef(null);

  // Get current chat messages
  const currentMessages = currentChatId ? (chats[currentChatId]?.messages || []) : [];

  // Scroll on new content
  useEffect(() => {
    timelineRef.current?.scrollTo({ top: timelineRef.current.scrollHeight, behavior: 'smooth' });
  }, [currentMessages, loading]);

  // Save chats to storage whenever they change
  useEffect(() => {
    saveChatsToStorage(chats);
  }, [chats]);

  // Create new chat
  const createNewChat = () => {
    const newChatId = Date.now().toString();
    const newChat = {
      id: newChatId,
      title: 'New Chat',
      messages: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    
    setChats(prev => ({ ...prev, [newChatId]: newChat }));
    setCurrentChatId(newChatId);
    setQuery('');
  };

  // Update chat title based on first user message
  const updateChatTitle = (chatId, firstMessage) => {
    const title = firstMessage.length > 50 ? firstMessage.substring(0, 50) + '...' : firstMessage;
    setChats(prev => ({
      ...prev,
      [chatId]: {
        ...prev[chatId],
        title,
        updatedAt: new Date().toISOString(),
      }
    }));
  };

  // Delete chat
  const deleteChat = (chatId) => {
    setChats(prev => {
      const newChats = { ...prev };
      delete newChats[chatId];
      return newChats;
    });
    
    if (currentChatId === chatId) {
      const remainingChats = Object.keys(chats).filter(id => id !== chatId);
      setCurrentChatId(remainingChats.length > 0 ? remainingChats[0] : null);
    }
  };

  // Handle source ranking submission
  const handleSourceRanking = async () => {
    try {
      const sourceRankingData = {
        training_id: currentTrainingId,
        sources: selectedSources.map(source => ({
          title: source.title,
          url: source.url,
          ranked: true
        }))
      };

      const response = await fetch('http://localhost:8000/save_source_ranking', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(sourceRankingData)
      });

      if (response.ok) {
        // Hide source selection and clear state
        setShowSourceSelection(false);
        setSelectedSources([]);
        setCurrentTrainingId(null);
        alert('Source rankings saved successfully!');
      } else {
        throw new Error('Failed to save source rankings');
      }
    } catch (error) {
      console.error('Error saving source rankings:', error);
      alert('Failed to save source rankings. Please try again.');
    }
  };

  // Handle training response selection
  const handleTrainingSelection = async (selectedResponse, comparisonData) => {
    try {
      // Prepare training data with sources
      const trainingData = {
        query: comparisonData.query || "Unknown query",
        response1: {
          text: comparisonData.response1.answer,
          temperature: comparisonData.response1.temperature,
          latency: comparisonData.response1.latency,
          tokens_per_minute: comparisonData.response1.tokens_per_minute
        },
        response2: {
          text: comparisonData.response2.answer,
          temperature: comparisonData.response2.temperature,
          latency: comparisonData.response2.latency,
          tokens_per_minute: comparisonData.response2.tokens_per_minute
        },
        selected_response: selectedResponse,
        sources: comparisonData.sources.map(source => ({
          title: source.title,
          url: source.url,
          ranked: false
        }))
      };

      // Send to backend
      const response = await fetch('http://localhost:8000/save_training_data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(trainingData)
      });

      if (response.ok) {
        const result = await response.json();
        const trainingId = result.training_id;
        
        // Update the chat to keep only the selected response
        setChats(prev => {
          const copy = { ...prev };
          const currentChat = copy[currentChatId];
          
          // Find the selected training message
          const selectedMessageIndex = currentChat.messages.findIndex(
            msg => msg.role === 'bot' && msg.training && msg.response === selectedResponse
          );
          
          if (selectedMessageIndex !== -1) {
            // Get the selected message
            const selectedMessage = currentChat.messages[selectedMessageIndex];
            
            // Convert it to a regular bot message (remove training properties)
            const regularMessage = {
              role: 'bot',
              content: selectedMessage.content,
              sources: selectedMessage.sources,
              mode: selectedMessage.mode,
              // Remove training-specific properties
              training: false,
              response: undefined,
              comparison: undefined,
              // Store training ID for source ranking
              trainingId: trainingId
            };
            
            // Remove all training messages and replace with the selected one
            const filteredMessages = currentChat.messages.filter(
              msg => !(msg.role === 'bot' && msg.training)
            );
            
            // Add the converted message at the end
            filteredMessages.push(regularMessage);
            
            copy[currentChatId] = {
              ...currentChat,
              messages: filteredMessages
            };
          }
          
          return copy;
        });

        // Show success message
        alert(`Response ${selectedResponse} selected as better! Training data saved.`);
        
        // Show source selection if there are sources
        if (comparisonData.sources && comparisonData.sources.length > 0) {
          setCurrentTrainingId(trainingId);
          setSelectedSources([]);
          setShowSourceSelection(true);
        }
      } else {
        throw new Error('Failed to save training data');
      }
    } catch (error) {
      console.error('Error saving training data:', error);
      alert('Failed to save training data. Please try again.');
    }
  };

  const submit = async () => {
    if (!query.trim() || loading || !currentChatId) return;

    const q = query;
    const m = mode;
    setQuery('');
    setLoading(true);
    setError('');

    // Add user message
    const updatedMessages = [...currentMessages, { role: 'user', content: q }];
    setChats(prev => ({
      ...prev,
      [currentChatId]: {
        ...prev[currentChatId],
        messages: updatedMessages,
        updatedAt: new Date().toISOString(),
      }
    }));

    // Update title if this is the first message
    if (updatedMessages.length === 1) {
      updateChatTitle(currentChatId, q);
    }

    // Add placeholder bot bubble(s)
    const botIndex = updatedMessages.length;
    if (training) {
      // Training mode: Add two placeholder responses
      setChats(prev => ({
        ...prev,
        [currentChatId]: {
          ...prev[currentChatId],
          messages: [...updatedMessages, 
            { role: 'bot', content: '', sources: [], mode: m, training: true, response: 1 },
            { role: 'bot', content: '', sources: [], mode: m, training: true, response: 2 }
          ],
        }
      }));
    } else {
      // Production mode: Single response
      setChats(prev => ({
        ...prev,
        [currentChatId]: {
          ...prev[currentChatId],
          messages: [...updatedMessages, { role: 'bot', content: '', sources: [], mode: m }],
        }
      }));
    }

    try {
      for await (const evt of await askStream(q, m, model, 5, training)) {
        if (evt.type === 'sources') {
          setChats(prev => {
            const copy = { ...prev };
            if (training) {
              copy[currentChatId].messages[botIndex].sources = evt.data;
              copy[currentChatId].messages[botIndex + 1].sources = evt.data;
            } else {
              copy[currentChatId].messages[botIndex].sources = evt.data;
            }
            return copy;
          });
        } else if (evt.type === 'token') {
          setChats(prev => {
            const copy = { ...prev };
            if (training && evt.response) {
              // Add duplicate detection for training mode
              const targetIndex = botIndex + evt.response - 1;
              const currentContent = copy[currentChatId].messages[targetIndex].content;
              if (currentContent.endsWith(evt.value)) {
                console.warn('Duplicate token detected in training mode:', evt.value, 'response:', evt.response);
                return copy; // Skip adding duplicate
              }
              copy[currentChatId].messages[targetIndex].content += evt.value;
            } else {
              // Add debugging to check for duplicate tokens
              const currentContent = copy[currentChatId].messages[botIndex].content;
              if (currentContent.endsWith(evt.value)) {
                console.warn('Duplicate token detected:', evt.value);
                return copy; // Skip adding duplicate
              }
              copy[currentChatId].messages[botIndex].content += evt.value;
            }
            return copy;
          });
        } else if (evt.type === 'training_comparison') {
          // Store comparison data for selection on both training messages
          setChats(prev => {
            const copy = { ...prev };
            copy[currentChatId].messages[botIndex].comparison = evt;
            copy[currentChatId].messages[botIndex + 1].comparison = evt;
            return copy;
          });
        } else if (evt.type === 'end') {
          setChats(prev => {
            const copy = { ...prev };
            if (training) {
              copy[currentChatId].messages[botIndex].content = evt.response1.answer;
              copy[currentChatId].messages[botIndex + 1].content = evt.response2.answer;
            }
            // In production mode, don't overwrite the content since it was already streamed
            // The content is already accumulated from the streaming tokens
            return copy;
          });
        }
      }
    } catch (e) {
      setError(e.message || 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  // Auto-create first chat if none exists
  useEffect(() => {
    if (Object.keys(chats).length === 0) {
      createNewChat();
    } else if (!currentChatId) {
      const chatIds = Object.keys(chats);
      setCurrentChatId(chatIds[0]);
    }
  }, []);

  return (
    <div className="h-screen flex bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-80' : 'w-16'} bg-white border-r border-slate-200 flex flex-col transition-all duration-300 ease-in-out`}>
        {/* Sidebar Header */}
        <div className="p-4 border-b border-slate-200">
          <div className="flex items-center justify-between">
            <div className={`flex items-center space-x-3 ${!sidebarOpen && 'hidden'}`}>
              <div className="w-8 h-8 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M9.99 0C4.47 0 0 4.48 0 10s4.47 10 9.99 10C15.52 20 20 15.52 20 10S15.52 0 9.99 0zM10 18c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm3.5-9c.83 0 1.5-.67 1.5-1.5S14.33 6 13.5 6 12 6.67 12 7.5s.67 1.5 1.5 1.5zm-7 0c.83 0 1.5-.67 1.5-1.5S7.33 6 6.5 6 5 6.67 5 7.5 5.67 9 6.5 9zm3.5 6.5c2.33 0 4.31-1.46 5.11-3.5H4.89c.8 2.04 2.78 3.5 5.11 3.5z" clipRule="evenodd" />
                </svg>
              </div>
              <div>
                <h1 className="text-lg font-bold gradient-text">Zerplixity</h1>
                <p className="text-xs text-slate-500">AI Search Assistant</p>
              </div>
            </div>
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-lg hover:bg-slate-100 transition-colors"
            >
              <svg className="w-5 h-5 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>

        {/* New Chat Button */}
        <div className="p-4">
          <button
            onClick={createNewChat}
            className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white px-4 py-2.5 rounded-lg font-medium hover:from-blue-700 hover:to-purple-700 transition-all duration-200 shadow-sm hover:shadow-md flex items-center justify-center space-x-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            {sidebarOpen && <span>New Chat</span>}
          </button>
        </div>

        {/* Chat List */}
        <div className="flex-1 overflow-y-auto">
          <div className="space-y-1 px-4">
            {Object.values(chats)
              .sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt))
              .map((chat) => (
                <div
                  key={chat.id}
                  className={`group sidebar-chat-item ${
                    currentChatId === chat.id
                      ? 'active'
                      : 'hover:bg-slate-50 border border-transparent'
                  }`}
                  onClick={() => setCurrentChatId(chat.id)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1 min-w-0">
                      <h3 className="text-sm font-medium text-slate-900 truncate">
                        {chat.title}
                      </h3>
                      <p className="text-xs text-slate-500 mt-1">
                        {new Date(chat.updatedAt).toLocaleDateString()}
                      </p>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteChat(chat.id);
                      }}
                      className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-100 transition-all duration-200"
                    >
                      <svg className="w-3 h-3 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  </div>
                </div>
              ))}
          </div>
              </div>
            </div>
            
      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="glass border-b border-slate-200 shadow-sm">
          <div className="px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <h2 className="text-xl font-semibold text-slate-800">
                  {currentChatId && chats[currentChatId] ? chats[currentChatId].title : 'New Chat'}
                </h2>
                <div className="flex items-center space-x-1 text-xs text-slate-500">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span>Online</span>
              </div>
              </div>
              
              {/* Training Mode Toggle */}
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-medium text-slate-700">Mode:</span>
                  <div className="flex bg-slate-100 rounded-lg p-1">
                    <button
                      onClick={() => setTraining(false)}
                      className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                        !training
                          ? 'bg-white text-slate-900 shadow-sm'
                          : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      Production
                    </button>
                    <button
                      onClick={() => setTraining(true)}
                      className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                        training
                          ? 'bg-white text-slate-900 shadow-sm'
                          : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      Training
                    </button>
                  </div>
                </div>
            </div>
          </div>
        </div>
      </header>

        {/* Chat Area */}
      <main
        ref={timelineRef}
          className="flex-1 overflow-y-auto px-6 py-6"
      >
        <div className="max-w-4xl mx-auto">
            {currentMessages.length === 0 && (
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-gradient-to-r from-blue-100 to-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-slate-800 mb-2">Welcome to Zerplixity</h2>
              <p className="text-slate-600 max-w-md mx-auto">
                Ask me anything and I'll search the web to find the most relevant and up-to-date information for you.
              </p>
                {training && (
                  <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <p className="text-sm text-yellow-800">
                      <strong>Training Mode:</strong> You'll see two model responses and can select the better one.
                    </p>
                  </div>
                )}
            </div>
          )}
          
          <div className="space-y-6">
              {currentMessages.map((m, idx) =>
              m.role === 'user' ? (
                <div key={idx} className="message-enter">
                  <UserBubble text={m.content} />
                </div>
              ) : (
                <div key={idx} className="message-enter">
                    {m.training ? (
                      <TrainingBubble 
                        text={m.content} 
                        sources={m.sources} 
                        mode={m.mode} 
                        response={m.response}
                        comparison={m.comparison}
                        onSelection={handleTrainingSelection}
                      />
                    ) : (
                  <BotBubble text={m.content} sources={m.sources} mode={m.mode} />
                    )}
                </div>
              )
            )}
            {loading && (
              <div className="message-enter">
                <div className="flex justify-start">
                  <div className="max-w-2xl lg:max-w-3xl">
                    <div className="bg-white border border-slate-200 px-6 py-4 rounded-2xl rounded-bl-md shadow-lg">
                      <div className="flex items-center space-x-2">
                        <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                          <span className="text-slate-600">
                            {training ? 'Generating responses for comparison...' : 'Searching for answers...'}
                          </span>
                      </div>
                    </div>
                    <div className="flex items-center mt-2 space-x-1">
                      <div className="w-6 h-6 bg-gradient-to-r from-green-400 to-blue-500 rounded-full flex items-center justify-center">
                        <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-6-3a2 2 0 11-4 0 2 2 0 014 0zm-2 4a5 5 0 00-4.546 2.916A5.986 5.986 0 0010 16a5.986 5.986 0 004.546-2.084A5 5 0 0010 11z" clipRule="evenodd" />
                        </svg>
                      </div>
                      <span className="text-xs text-slate-500">AI Assistant</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Input Area */}
      <div className="glass border-t border-slate-200 shadow-lg">
          <div className="px-6 py-4">
          <ChatInput
            value={query}
            onChange={setQuery}
            onSubmit={submit}
            loading={loading}
            mode={mode}
            onModeChange={setMode}
            model={model} 
            onModelChange={setModel}
              disabled={!currentChatId}
          />
        </div>
      </div>

      {/* Error Message */}
      {error && (
          <div className="bg-red-50 border-t border-red-200 px-6 py-3">
            <div className="flex items-center space-x-2 text-red-700">
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            <span className="text-sm font-medium">{error}</span>
          </div>
        </div>
      )}

        {/* Source Selection Modal */}
        {showSourceSelection && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-slate-900">Select Top 3 Most Relevant Sources</h3>
                <button
                  onClick={() => setShowSourceSelection(false)}
                  className="text-slate-400 hover:text-slate-600"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <p className="text-sm text-slate-600 mb-4">
                Help us improve our source ranking by selecting the top 3 most relevant sources for this query.
              </p>
              
              <div className="space-y-3 mb-6">
                {currentMessages
                  .filter(msg => msg.role === 'bot' && msg.trainingId === currentTrainingId)
                  .flatMap(msg => msg.sources || [])
                  .map((source, index) => (
                    <div key={index} className="flex items-start space-x-3 p-3 border border-slate-200 rounded-lg hover:bg-slate-50">
                      <input
                        type="checkbox"
                        id={`source-${index}`}
                        checked={selectedSources.some(s => s.url === source.url)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            if (selectedSources.length < 3) {
                              setSelectedSources([...selectedSources, source]);
                            }
                          } else {
                            setSelectedSources(selectedSources.filter(s => s.url !== source.url));
                          }
                        }}
                        className="mt-1 h-4 w-4 text-blue-600 focus:ring-blue-500 border-slate-300 rounded"
                      />
                      <div className="flex-1">
                        <label htmlFor={`source-${index}`} className="block text-sm font-medium text-slate-900 cursor-pointer">
                          {source.title}
                        </label>
                        <p className="text-xs text-slate-500 mt-1">{source.url}</p>
                      </div>
                    </div>
                  ))}
              </div>
              
              <div className="flex items-center justify-between">
                <p className="text-sm text-slate-600">
                  Selected: {selectedSources.length}/3 sources
                </p>
                <div className="flex space-x-3">
                  <button
                    onClick={() => setShowSourceSelection(false)}
                    className="px-4 py-2 text-sm font-medium text-slate-700 bg-slate-100 rounded-md hover:bg-slate-200 transition-colors"
                  >
                    Skip
                  </button>
                  <button
                    onClick={handleSourceRanking}
                    disabled={selectedSources.length === 0}
                    className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    Save Rankings
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------- Bubbles ---------- */

function UserBubble({ text }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-2xl lg:max-w-3xl">
        <div className="bg-gradient-to-r from-blue-600 to-blue-700 text-white px-6 py-4 rounded-2xl rounded-br-md shadow-lg">
          <div className="text-sm leading-relaxed">{text}</div>
        </div>
        <div className="flex justify-end mt-2">
          <div className="flex items-center space-x-1 text-xs text-slate-500">
            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            <span>You</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function BotBubble({ text, sources = [], mode }) {
  return (
    <div className="flex justify-start">
      <div className="max-w-2xl lg:max-w-3xl">
        <div className="bg-white border border-slate-200 px-6 py-4 rounded-2xl rounded-bl-md shadow-lg">
          {mode && (
            <div className="mb-3">
              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                mode === 'vanilla' 
                  ? 'bg-blue-100 text-blue-800' 
                  : 'bg-purple-100 text-purple-800'
              }`}>
                {mode === 'vanilla' ? 'Vanilla Retrieval' : 'Reranker Retrieval'}
              </span>
            </div>
          )}
          <RenderAnswer text={text} sources={sources} />
        </div>
        <div className="flex items-center mt-2 space-x-1">
          <div className="w-6 h-6 bg-gradient-to-r from-green-400 to-blue-500 rounded-full flex items-center justify-center">
            <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-6-3a2 2 0 11-4 0 2 2 0 014 0zm-2 4a5 5 0 00-4.546 2.916A5.986 5.986 0 0010 16a5.986 5.986 0 004.546-2.084A5 5 0 0010 11z" clipRule="evenodd" />
            </svg>
          </div>
          <span className="text-xs text-slate-500">AI Assistant</span>
        </div>
      </div>
    </div>
  );
}

function TrainingBubble({ text, sources = [], mode, response, comparison, onSelection }) {
  const isResponse1 = response === 1;
  const isResponse2 = response === 2;
  
  return (
    <div className="flex justify-start">
      <div className="max-w-2xl lg:max-w-3xl">
        <div className={`border border-slate-200 px-6 py-4 rounded-2xl rounded-bl-md shadow-lg ${
          isResponse1 ? 'bg-blue-50 border-blue-200' : 'bg-purple-50 border-purple-200'
        }`}>
          {/* Response Header */}
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                isResponse1 
                  ? 'bg-blue-100 text-blue-800' 
                  : 'bg-purple-100 text-purple-800'
              }`}>
                Response {response}
              </span>
              {mode && (
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                  mode === 'vanilla' 
                    ? 'bg-gray-100 text-gray-800' 
                    : 'bg-orange-100 text-orange-800'
                }`}>
                  {mode === 'vanilla' ? 'Vanilla' : 'Rerank'}
                </span>
              )}
            </div>
            
            {/* Metrics */}
            {comparison && (
              <div className="flex items-center space-x-3 text-xs text-slate-500">
                <span>Latency: {comparison[`response${response}`].latency}s</span>
                <span>Speed: {comparison[`response${response}`].tokens_per_minute} tokens/min</span>
                <span>Temp: {comparison[`response${response}`].temperature}</span>
              </div>
            )}
          </div>
          
          {/* Response Content */}
          <RenderAnswer text={text} sources={sources} />
          
          {/* Selection Buttons */}
          {comparison && (
            <div className="mt-4 pt-3 border-t border-slate-200">
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-600">Select this response as better:</span>
                <div className="flex space-x-2">
                  <button
                    onClick={() => onSelection(response, comparison)}
                    className="px-3 py-1 bg-green-100 text-green-700 text-xs font-medium rounded-md hover:bg-green-200 transition-colors"
                  >
                    Select Better
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
        
        <div className="flex items-center mt-2 space-x-1">
          <div className={`w-6 h-6 rounded-full flex items-center justify-center ${
            isResponse1 
              ? 'bg-gradient-to-r from-blue-400 to-blue-500' 
              : 'bg-gradient-to-r from-purple-400 to-purple-500'
          }`}>
            <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-6-3a2 2 0 11-4 0 2 2 0 014 0zm-2 4a5 5 0 00-4.546 2.916A5.986 5.986 0 0010 16a5.986 5.986 0 004.546-2.084A5 5 0 0010 11z" clipRule="evenodd" />
            </svg>
          </div>
          <span className="text-xs text-slate-500">AI Assistant (Response {response})</span>
        </div>
      </div>
    </div>
  );
}

/* ---------- Input ---------- */

function ChatInput({ value, onChange, onSubmit, loading, mode, onModeChange, model, onModelChange, disabled }) {
  return (
    <div className="max-w-4xl mx-auto">
    <div className="flex flex-col sm:flex-row gap-3">
      {/* Mode Selector */}
      <div className="relative">
        <select
            className="appearance-none bg-white border border-slate-300 rounded-lg px-3 py-2.5 pr-8 text-sm font-medium text-slate-700 input-focus disabled:opacity-50 disabled:cursor-not-allowed min-w-[140px]"
            disabled={loading || disabled}
          value={mode}
          onChange={e => onModeChange(e.target.value)}
        >
          <option value="vanilla">Vanilla Retrieval</option>
          <option value="rerank">Reranker Retrieval</option>
        </select>
        <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
          <svg className="w-4 h-4 text-slate-400" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
          </svg>
        </div>
      </div>

        {/* Model Selector */}
      <div className="relative">
        <select
            className="appearance-none bg-white border border-slate-300 rounded-lg px-3 py-2.5 pr-8 text-sm font-medium text-slate-700 input-focus disabled:opacity-50 min-w-[140px]"
            disabled={loading || disabled}
          value={model}
          onChange={e => onModelChange(e.target.value)}
        >
            <option value="gpt4o">GPT‑4o mini</option>
            <option value="llama">Llama 80B</option>
        </select>
        <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
          <svg className="w-4 h-4 text-slate-400" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
          </svg>
        </div>
      </div>

      {/* Input Field */}
      <div className="flex-1 relative">
        <input
            className="w-full bg-white border border-slate-300 rounded-lg px-4 py-2.5 pr-12 text-sm placeholder-slate-400 input-focus disabled:opacity-50"
          placeholder="Ask me anything about the world..."
          value={value}
          onChange={e => onChange(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && onSubmit()}
            disabled={loading || disabled}
        />
        <div className="absolute inset-y-0 right-0 flex items-center pr-3">
          <svg className="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
      </div>

      {/* Send Button */}
              <button
          onClick={onSubmit}
          disabled={loading || !value.trim() || disabled}
          className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed px-6 py-2.5"
        >
        {loading ? (
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            <span>Searching...</span>
          </div>
        ) : (
          <div className="flex items-center space-x-2">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
            <span>Send</span>
          </div>
        )}
      </button>
      </div>
    </div>
  );
}
