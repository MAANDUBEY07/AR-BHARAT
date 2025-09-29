import React, { useState } from "react";
import axios from "axios";
import { Link } from "react-router-dom";

export default function UploadPage(){
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [pngSrc, setPngSrc] = useState(null);
  const [svgSrc, setSvgSrc] = useState(null);
  const [meta, setMeta] = useState(null);
  const [patternId, setPatternId] = useState(null);
  const [precisionMode, setPrecisionMode] = useState('traditional'); // Default to traditional mode for better visuals
  async function upload(){
    if(!file) return alert("Choose an image");
    setLoading(true);
    const fd = new FormData();
    fd.append("file", file);
    fd.append("precision_mode", precisionMode); // Pass the selected mode
    
    try {
      // Use selected precision mode for generation
      const apiUrl = import.meta.env.PROD ? 'https://ar-bharat-1.onrender.com' : '';
      const res = await axios.post(`${apiUrl}/api/kolam-from-image`, fd, { timeout: 120000 });
      
      setSvgSrc("data:image/svg+xml;utf8," + encodeURIComponent(res.data.svg));
      setPngSrc(null);
      setMeta(res.data.metadata || null);
      setPatternId(res.data.id || null);
      
      // Success message
      const modeNames = {
        traditional: 'Traditional Flowing Kolam',
        enhanced: 'Enhanced Precision Kolam', 
        ultra: 'Ultra-Precision Kolam',
        basic: 'Basic Kolam'
      };
      alert(`${modeNames[precisionMode]} generated successfully!`);
    } catch(err){
      console.error(err);
      alert("Error processing image: " + (err.response?.data?.error || err.message));
    } finally {
      setLoading(false);
    }
  }

  // MATLAB-style Kolam generator
  const [matlabSvg, setMatlabSvg] = useState(null);
  const [kolamSize, setKolamSize] = useState(7);
  async function generateMatlabKolam() {
    setLoading(true);
    try {
      const payload = {
        type: '2d',
        size: parseInt(kolamSize),
        spacing: 40,
        color: '#2458ff',
        linewidth: 2,
        show_dots: true,
        artist: 'Web User'
      };
      const apiUrl = import.meta.env.PROD ? 'https://ar-bharat-1.onrender.com' : '';
      const res = await axios.post(`${apiUrl}/api/kolam-matlab`, payload, {
        headers: { 'Content-Type': 'application/json' }
      });
      setMatlabSvg(res.data.svg);
    } catch (err) {
      alert('Kolam generation failed: ' + (err.response?.data?.error || err.message));
    } finally {
      setLoading(false);
    }
  }
  return (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded shadow">
        <h2 className="text-xl font-semibold">Upload Rangoli / Kolam Image</h2>
        <input type="file" accept="image/*" onChange={e=>setFile(e.target.files[0])} className="mt-4" />
        
        {/* Precision Mode Selector */}
        <div className="mt-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">Generation Mode:</label>
          <select 
            value={precisionMode} 
            onChange={e=>setPrecisionMode(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            <option value="traditional">Traditional Flowing Kolam (Recommended)</option>
            <option value="enhanced">Enhanced Precision Kolam</option>
            <option value="ultra">Ultra-Precision Kolam (96% Accuracy)</option>
            <option value="basic">Basic Kolam</option>
          </select>
          <p className="text-xs text-gray-500 mt-1">
            Traditional mode creates beautiful flowing designs like traditional Tamil kolams
          </p>
        </div>
        
        <div className="mt-4 flex gap-4">
          <button onClick={upload} className="px-4 py-2 bg-indigo-600 text-white rounded font-medium" disabled={loading}>
            {loading ? "Processing..." : "Generate High-Quality Kolam"}
          </button>
          <input type="number" min={3} max={21} value={kolamSize} onChange={e=>setKolamSize(e.target.value)} className="px-2 py-1 border rounded w-20" />
          <button onClick={generateMatlabKolam} className="px-4 py-2 bg-green-600 text-white rounded" disabled={loading}>
            {loading ? "Processing..." : "Dot-Grid Kolam"}
          </button>
        </div>
        {svgSrc && (
          <div className="bg-white p-4 rounded shadow mt-6">
            <h3 className="font-medium">Generated Kolam Pattern</h3>
            {meta && (
              <div className="mb-3 p-3 bg-gray-50 rounded text-sm">
                <div className="flex gap-4">
                  <span><strong>Pattern:</strong> {meta.pattern_type || 'Traditional'}</span>
                  <span className="text-indigo-600"><strong>Quality:</strong> High-Precision Generated</span>
                </div>
                {meta.features && (
                  <div className="mt-2 text-xs text-gray-600">
                    Features: {Object.keys(meta.features).filter(k => meta.features[k]).join(', ')}
                  </div>
                )}
              </div>
            )}
            <div dangerouslySetInnerHTML={{__html: decodeURIComponent(svgSrc.split(',')[1] || '')}} className="mt-2 border p-2 bg-gray-50" />
            <div className="mt-3 flex space-x-3">
              <a href={svgSrc} download="kolam.svg" className="text-sm text-indigo-600 hover:text-indigo-800">Download SVG</a>
              {patternId && (
                <span className="text-xs text-gray-500">Pattern ID: {patternId}</span>
              )}
            </div>
          </div>
        )}
        {matlabSvg && (
          <div className="bg-white p-4 rounded shadow mt-6">
            <h3 className="font-medium">Dot-Grid Kolam Pattern</h3>
            <div dangerouslySetInnerHTML={{__html: matlabSvg}} className="mt-2 border p-2" />
            <div className="mt-3 flex space-x-3">
              <a href={"data:image/svg+xml;utf8," + encodeURIComponent(matlabSvg)} download="dot-grid-kolam.svg" className="text-sm text-green-600 hover:text-green-800">Download SVG</a>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}