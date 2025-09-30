import React, {useEffect, useState} from "react";
import axios from "axios";
import { Link } from "react-router-dom";

export default function GalleryPage(){
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(()=>{ load(); }, []);
  
  async function load(){
    try {
      setLoading(true);
      setError(null);
      const res = await axios.get("/api/patterns");
      // Ensure we have an array
      if (Array.isArray(res.data)) {
        setItems(res.data);
      } else if (res.data && Array.isArray(res.data.patterns)) {
        setItems(res.data.patterns);
      } else {
        // Fallback with demo patterns if API fails
        setItems([
          {
            id: 1,
            name: "Traditional Kolam Pattern",
            svg: '<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><circle cx="100" cy="100" r="80" stroke="#4F46E5" stroke-width="3" fill="none"/><circle cx="100" cy="100" r="40" stroke="#4F46E5" stroke-width="3" fill="none"/><circle cx="100" cy="100" r="20" stroke="#4F46E5" stroke-width="3" fill="none"/></svg>',
            created_at: new Date().toISOString()
          },
          {
            id: 2,
            name: "Geometric Rangoli",
            svg: '<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><polygon points="100,20 180,180 20,180" stroke="#7C3AED" stroke-width="3" fill="none"/><polygon points="100,60 140,140 60,140" stroke="#7C3AED" stroke-width="3" fill="none"/></svg>',
            created_at: new Date().toISOString()
          },
          {
            id: 3,
            name: "Mandala Design",
            svg: '<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><circle cx="100" cy="100" r="90" stroke="#EC4899" stroke-width="2" fill="none"/><circle cx="100" cy="100" r="60" stroke="#EC4899" stroke-width="2" fill="none"/><circle cx="100" cy="100" r="30" stroke="#EC4899" stroke-width="2" fill="none"/><line x1="100" y1="10" x2="100" y2="190" stroke="#EC4899" stroke-width="2"/><line x1="10" y1="100" x2="190" y2="100" stroke="#EC4899" stroke-width="2"/></svg>',
            created_at: new Date().toISOString()
          }
        ]);
      }
    } catch (err) {
      console.error('Gallery API Error:', err);
      setError('Failed to load gallery. Showing demo patterns.');
      // Fallback with demo patterns
      setItems([
        {
          id: 1,
          name: "Traditional Kolam Pattern",
          svg: '<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><circle cx="100" cy="100" r="80" stroke="#4F46E5" stroke-width="3" fill="none"/><circle cx="100" cy="100" r="40" stroke="#4F46E5" stroke-width="3" fill="none"/><circle cx="100" cy="100" r="20" stroke="#4F46E5" stroke-width="3" fill="none"/></svg>',
          created_at: new Date().toISOString()
        },
        {
          id: 2,
          name: "Geometric Rangoli",
          svg: '<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><polygon points="100,20 180,180 20,180" stroke="#7C3AED" stroke-width="3" fill="none"/><polygon points="100,60 140,140 60,140" stroke="#7C3AED" stroke-width="3" fill="none"/></svg>',
          created_at: new Date().toISOString()
        },
        {
          id: 3,
          name: "Mandala Design",  
          svg: '<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><circle cx="100" cy="100" r="90" stroke="#EC4899" stroke-width="2" fill="none"/><circle cx="100" cy="100" r="60" stroke="#EC4899" stroke-width="2" fill="none"/><circle cx="100" cy="100" r="30" stroke="#EC4899" stroke-width="2" fill="none"/><line x1="100" y1="10" x2="100" y2="190" stroke="#EC4899" stroke-width="2"/><line x1="10" y1="100" x2="190" y2="100" stroke="#EC4899" stroke-width="2"/></svg>',
          created_at: new Date().toISOString()
        }
      ]);
    } finally {
      setLoading(false);
    }
  }
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">Pattern Gallery</h1>
        <p className="text-xl text-gray-600 max-w-2xl mx-auto">
          Discover beautiful Kolam and Rangoli patterns created with AR BHARAT's AI technology
        </p>
      </div>

      {error && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
          <div className="flex">
            <div className="flex-shrink-0">
              <span className="text-yellow-400">⚠️</span>
            </div>
            <div className="ml-3">
              <p className="text-sm text-yellow-700">{error}</p>
            </div>
          </div>
        </div>
      )}

      {loading ? (
        <div className="flex justify-center items-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600"></div>
          <span className="ml-3 text-gray-600">Loading patterns...</span>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {items.map(it => (
          <div key={it.id} className="bg-white p-4 rounded shadow" data-testid="pattern-card">
            <div className="relative">
              {it.png ? (
                <img src={`/storage/${it.png}`} alt={it.name} className="w-full h-48 object-contain border rounded-lg" />
              ) : (
                <div className="w-full h-48 border rounded-lg flex items-center justify-center bg-gray-50 overflow-hidden relative">
                  <div 
                    dangerouslySetInnerHTML={{__html: it.svg || ''}} 
                    className="absolute inset-0 flex items-center justify-center p-2"
                    style={{
                      transform: 'scale(0.8)',
                      transformOrigin: 'center'
                    }}
                  />
                </div>
              )}
              <div className="absolute top-2 right-2">
                <Link 
                  to={`/ar/${it.id}`} 
                  className="bg-purple-500 hover:bg-purple-700 text-white px-3 py-1 rounded-full text-xs font-semibold transition-colors flex items-center space-x-1"
                >
                  <span>🥽</span>
                  <span>AR</span>
                </Link>
              </div>
            </div>
            <div className="mt-3">
              <div className="font-medium">{it.name}</div>
              <div className="text-xs text-slate-500 mb-3">{new Date(it.created_at).toLocaleString()}</div>
              
              <div className="flex justify-between items-center">
                <div className="flex space-x-2">
                  <a href={`/api/patterns/${it.id}/download?format=svg`} className="text-sm text-indigo-600 hover:text-indigo-800">SVG</a>
                  <a href={`/api/patterns/${it.id}/download?format=png`} className="text-sm text-indigo-600 hover:text-indigo-800">PNG (4K)</a>
                </div>
                <Link 
                  to={`/ar/${it.id}`} 
                  className="bg-gradient-to-r from-purple-500 to-indigo-600 text-white px-4 py-2 rounded-lg text-sm font-semibold hover:from-purple-600 hover:to-indigo-700 transition-all flex items-center space-x-2"
                >
                  <span>🚀</span>
                  <span>Launch AR</span>
                </Link>
              </div>
            </div>
          </div>
          ))}
        </div>
      )}

      {!loading && items.length === 0 && (
        <div className="text-center py-12">
          <div className="text-6xl mb-4">🎨</div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">No patterns yet</h3>
          <p className="text-gray-600 mb-6">Be the first to create beautiful Kolam patterns!</p>
          <Link 
            to="/upload" 
            className="bg-gradient-to-r from-purple-500 to-indigo-600 text-white px-6 py-3 rounded-lg font-semibold hover:from-purple-600 hover:to-indigo-700 transition-all"
          >
            Upload Your First Image
          </Link>
        </div>
      )}
    </div>
  );
}