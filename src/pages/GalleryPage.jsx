import React, {useEffect, useState} from "react";
import axios from "axios";
import { Link } from "react-router-dom";

export default function GalleryPage(){
  const [items, setItems] = useState([]);
  useEffect(()=>{ load(); }, []);
  async function load(){
    const res = await axios.get("/api/patterns");
    setItems(res.data);
  }
  return (
    <div>
      <h2 className="text-2xl font-semibold mb-4">Gallery</h2>
      <div className="grid md:grid-cols-3 gap-6">
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
    </div>
  );
}