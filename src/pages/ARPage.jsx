import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { useParams } from "react-router-dom";
import { Howl } from "howler";
import { ARButton } from "../utils/xr-button";

export default function ARPage(){
  const mountRef = useRef();
  const { id } = useParams();
  const [pngUrl, setPngUrl] = useState(null);
  const [audioPlaying, setAudioPlaying] = useState(false);
  const [pattern, setPattern] = useState(null);
  const [isARSupported, setIsARSupported] = useState(false);
  const [arActive, setArActive] = useState(false);

  useEffect(()=>{
    async function loadMeta(){
      const list = await fetch(`/api/patterns`).then(r=>r.json());
      const item = list.find(x=>x.id === parseInt(id));
      if(item){
        setPngUrl(`/storage/${item.png}`);
        setPattern(item);
      }
    }
    loadMeta();
    
    // Check AR support
    if (navigator.xr) {
      navigator.xr.isSessionSupported('immersive-ar').then(supported => {
        setIsARSupported(supported);
      });
    }
  }, [id]);

  useEffect(()=>{
    if(!pngUrl) return;
    const mount = mountRef.current;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(70, mount.clientWidth/mount.clientHeight, 0.01, 20);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.xr.enabled = true;
    mount.appendChild(renderer.domElement);

    const light = new THREE.HemisphereLight(0xffffff, 0x444444, 1.2);
    scene.add(light);

    const loader = new THREE.TextureLoader();
    loader.load(pngUrl, (tex) => {
      tex.flipY = true;
      const aspect = tex.image.width / tex.image.height;
      const width = 1.5;
      const height = width / aspect;
      const geo = new THREE.PlaneGeometry(width, height);
      const mat = new THREE.MeshStandardMaterial({ map: tex, transparent: true });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.rotation.x = -Math.PI/2;
      mesh.position.set(0, 0.01, -0.5);
      scene.add(mesh);
    });

    camera.position.set(0, 1.6, 1.5);

    function animate() { renderer.setAnimationLoop(()=>{ renderer.render(scene, camera); }); }
    animate();

    // XRButton: only add if available
    let xrButton;
    if (navigator.xr && isARSupported) {
      xrButton = ARButton.createButton(renderer);
      xrButton.addEventListener('click', () => {
        setArActive(true);
      });
      mount.appendChild(xrButton);
    }

    // cleanup
    return ()=>{ 
      renderer.dispose(); 
      if(mount && renderer.domElement && mount.contains(renderer.domElement)) {
        mount.removeChild(renderer.domElement); 
      }
      if(xrButton && mount && mount.contains(xrButton)) {
        mount.removeChild(xrButton); 
      }
    }
  }, [pngUrl, isARSupported]);

  // Howler narration (simple auto-generated text -> TTS would go here; we include a placeholder sound)
  async function playNarration(){
    // For demo use a short recorded narration file or TTS output URL.
    // Here we expect a server-side endpoint /api/narration/<id>.mp3 (you can generate TTS and store)
    const url = `/api/patterns/${id}/narration.mp3`; // create this in backend or provide static file
    const sound = new Howl({ src: [url], html5: true });
    sound.play();
    setAudioPlaying(true);
    sound.on('end', ()=> setAudioPlaying(false));
  }

  return (
    <div className="space-y-6">
      {/* AR Experience Header */}
      <div className="bg-gradient-to-r from-purple-500 to-indigo-600 text-white p-6 rounded-lg shadow-lg">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold mb-2">🥽 AR Viewer</h1>
            <p className="text-purple-100">
              {pattern ? `Viewing: ${pattern.name}` : 'Loading pattern...'}
            </p>
          </div>
          <div className="text-right">
            <div className="text-sm text-purple-200">AR Status</div>
            <div className="font-semibold" data-testid="ar-status">
              {arActive ? 'Active' : isARSupported ? 'Ready' : 'Not Available'}
            </div>
          </div>
        </div>
      </div>

      {/* AR Viewer */}
      <div className="bg-white p-6 rounded-lg shadow-lg" data-testid="ar-viewer">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">3D Preview</h3>
          {!isARSupported && (
            <div className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-sm">
              AR not supported on this device
            </div>
          )}
        </div>
        
        <div ref={mountRef} style={{ height: 520 }} className="bg-gray-100 rounded-lg overflow-hidden border" data-testid="3d-preview" />
        
        <div className="mt-4 flex flex-wrap gap-3">
          <button 
            onClick={playNarration} 
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors flex items-center space-x-2"
          >
            <span>{audioPlaying ? "🔊" : "🎵"}</span>
            <span>{audioPlaying ? "Playing..." : "Play Narration"}</span>
          </button>
          
          <button 
            className="px-4 py-2 border-2 border-indigo-600 text-indigo-600 rounded-lg hover:bg-indigo-50 transition-colors flex items-center space-x-2"
          >
            <span>📦</span>
            <span>Export 3D Model</span>
          </button>
          
          <button 
            onClick={() => navigator.share && pattern && navigator.share({
              title: `AR Kolam Pattern: ${pattern.name}`,
              text: 'Check out this amazing AR Kolam pattern!',
              url: window.location.href
            })}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors flex items-center space-x-2"
          >
            <span>📤</span>
            <span>Share AR</span>
          </button>
        </div>
      </div>

      {/* AR Instructions */}
      <div className="bg-blue-50 p-4 rounded-lg border-l-4 border-blue-400" data-testid="ar-instructions">
        <h4 className="font-semibold text-blue-900 mb-2">🚀 How to use AR:</h4>
        <div className="space-y-2 text-blue-800">
          {isARSupported ? (
            <>
              <p>• Tap the "Enter AR" button in the 3D viewer above</p>
              <p>• Point your camera at a flat surface (floor, table, etc.)</p>
              <p>• The Kolam pattern will appear on the surface in real size</p>
              <p>• Move around to view the pattern from different angles</p>
              <p>• Perfect for tracing or understanding the full pattern layout!</p>
            </>
          ) : (
            <>
              <p>• AR requires a mobile device with AR support (ARCore/ARKit)</p>
              <p>• Try accessing this page on a modern smartphone or tablet</p>
              <p>• Use Chrome or Safari for the best AR experience</p>
              <p>• For now, enjoy the 3D preview above!</p>
            </>
          )}
        </div>
      </div>

      {/* Pattern Information */}
      {pattern && (
        <div className="bg-white p-4 rounded-lg shadow" data-testid="pattern-info">
          <h4 className="font-semibold mb-3">Pattern Details</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Name:</span>
              <div className="font-medium">{pattern.name}</div>
            </div>
            <div>
              <span className="text-gray-500">Created:</span>
              <div className="font-medium">{new Date(pattern.created_at).toLocaleDateString()}</div>
            </div>
            <div>
              <span className="text-gray-500">Format:</span>
              <div className="font-medium">PNG + SVG</div>
            </div>
            <div>
              <span className="text-gray-500">Quality:</span>
              <div className="font-medium">4K Ultra HD</div>
            </div>
          </div>
          <div className="mt-4 flex space-x-3">
            <a 
              href={`/api/patterns/${pattern.id}/download?format=png`} 
              className="text-indigo-600 hover:text-indigo-800 text-sm font-medium"
            >
              Download PNG (4K)
            </a>
            <a 
              href={`/api/patterns/${pattern.id}/download?format=svg`} 
              className="text-indigo-600 hover:text-indigo-800 text-sm font-medium"
            >
              Download SVG
            </a>
          </div>
        </div>
      )}
    </div>
  );
}