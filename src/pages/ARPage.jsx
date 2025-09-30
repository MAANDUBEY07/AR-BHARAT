import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { useParams } from "react-router-dom";
import { motion } from "framer-motion";
import { Howl } from "howler";
import { ARButton } from "../utils/xr-button";

// Monument data for AR Heritage Experience
const MONUMENTS = [
  {
    id: 1,
    name: "Taj Mahal",
    icon: "🕌",
    location: "Agra, UP",
    year: "1653 CE",
    description: "Timeless symbol of love and Mughal architecture",
    arFeatures: ["360° View", "Historical Timeline", "Architectural Details"],
    color: "from-pink-500 to-rose-600"
  },
  {
    id: 2,
    name: "Red Fort",
    icon: "🏰",
    location: "Delhi",
    year: "1648 CE", 
    description: "Majestic Mughal fortress and seat of power",
    arFeatures: ["Virtual Walkthrough", "Court Scenes", "Battle History"],
    color: "from-red-500 to-red-700"
  },
  {
    id: 3,
    name: "Konark Temple",
    icon: "🛕",
    location: "Odisha",
    year: "1250 CE",
    description: "Sun Temple with magnificent stone chariot design",
    arFeatures: ["Solar Calendar", "Stone Sculptures", "Celestial Alignment"],
    color: "from-orange-500 to-amber-600"
  },
  {
    id: 4,
    name: "Gateway of India",
    icon: "🏛",
    location: "Mumbai",
    year: "1924 CE",
    description: "Iconic colonial monument overlooking Arabian Sea",
    arFeatures: ["Harbor Views", "Colonial History", "Independence Stories"],
    color: "from-blue-500 to-cyan-600"
  },
  {
    id: 5,
    name: "Hawa Mahal",
    icon: "👑",
    location: "Jaipur",
    year: "1799 CE",
    description: "Palace of Winds with intricate pink sandstone facade",
    arFeatures: ["Wind System", "Royal Chambers", "Pink City Views"],
    color: "from-pink-400 to-pink-600"
  }
];

export default function ARPage() {
  const mountRef = useRef();
  const { id } = useParams();
  const [selectedMonument, setSelectedMonument] = useState(null);
  const [currentView, setCurrentView] = useState('heritage'); // 'heritage' | 'kolam' | 'monument'
  const [pngUrl, setPngUrl] = useState(null);
  const [audioPlaying, setAudioPlaying] = useState(false);
  const [pattern, setPattern] = useState(null);
  const [isARSupported, setIsARSupported] = useState(false);
  const [arActive, setArActive] = useState(false);
  const [kolamARActive, setKolamARActive] = useState(true);

  useEffect(() => {
    // Check AR support
    if (navigator.xr) {
      navigator.xr.isSessionSupported('immersive-ar').then(supported => {
        setIsARSupported(supported);
      });
    }
    
    // If an ID is provided, load the Kolam pattern
    if (id) {
      setCurrentView('kolam');
      loadKolamPattern();
    } else {
      setCurrentView('heritage');
    }
  }, [id]);

  async function loadKolamPattern() {
    try {
      const list = await fetch(`/api/patterns`).then(r => r.json());
      const item = list.find(x => x.id === parseInt(id));
      if (item) {
        setPngUrl(`/storage/${item.png}`);
        setPattern(item);
      }
    } catch (error) {
      console.error('Failed to load pattern:', error);
    }
  }

  // Three.js AR Setup for Kolam patterns
  useEffect(() => {
    if (currentView !== 'kolam' || !pngUrl) return;
    
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(70, mount.clientWidth / mount.clientHeight, 0.01, 20);
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
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.set(0, 0.01, -0.5);
      scene.add(mesh);
    });

    camera.position.set(0, 1.6, 1.5);

    function animate() {
      renderer.setAnimationLoop(() => { renderer.render(scene, camera); });
    }
    animate();

    // XRButton
    let xrButton;
    if (navigator.xr && isARSupported) {
      xrButton = ARButton.createButton(renderer);
      xrButton.addEventListener('click', () => {
        setArActive(true);
      });
      mount.appendChild(xrButton);
    }

    // Cleanup
    return () => {
      renderer.dispose();
      if (mount && renderer.domElement && mount.contains(renderer.domElement)) {
        mount.removeChild(renderer.domElement);
      }
      if (xrButton && mount && mount.contains(xrButton)) {
        mount.removeChild(xrButton);
      }
    };
  }, [pngUrl, isARSupported, currentView]);

  async function playNarration() {
    const sound = new Howl({ src: [`/api/patterns/${id}/narration.mp3`], html5: true });
    sound.play();
    setAudioPlaying(true);
    sound.on('end', () => setAudioPlaying(false));
  }

  const handleMonumentSelect = (monument) => {
    setSelectedMonument(monument);
    setCurrentView('monument');
  };

  const renderHeritageExperience = () => (
    <div className="space-y-8">
      {/* AR Heritage Experience Header */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-center space-y-4"
      >
        <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-4">
          AR Heritage Experience
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Immerse yourself in augmented reality cultural experiences
        </p>
        
        {/* AR Status Indicator */}
        <motion.div 
          initial={{ scale: 0.9 }}
          animate={{ scale: 1 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className={`inline-flex items-center px-6 py-3 rounded-full text-white font-semibold ${
            kolamARActive ? 'bg-green-500' : 'bg-gray-500'
          }`}
        >
          <span className="text-2xl mr-3">🪷</span>
          <span>Kolam AR Mode Active</span>
        </motion.div>
        
        <p className="text-gray-600">
          Traditional patterns overlaid on floor surface
        </p>
      </motion.div>

      {/* Monument Selection Grid */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.4 }}
        className="space-y-6"
      >
        <div className="text-center">
          <h2 className="text-3xl font-bold text-gray-800 mb-2">Select Your AR Heritage Journey</h2>
          <p className="text-gray-600">Choose a monument to explore in augmented reality</p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {MONUMENTS.map((monument, index) => (
            <motion.div
              key={monument.id}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.1 * index }}
              whileHover={{ scale: 1.02 }}
              onClick={() => handleMonumentSelect(monument)}
              className={`bg-gradient-to-br ${monument.color} p-6 rounded-2xl shadow-lg text-white cursor-pointer transform transition-all duration-300 hover:shadow-xl`}
            >
              <div className="text-center space-y-4">
                <div className="text-6xl mb-4">{monument.icon}</div>
                <h3 className="text-2xl font-bold">{monument.name}</h3>
                <div className="space-y-2 text-sm opacity-90">
                  <p>📍 {monument.location}</p>
                  <p>📅 {monument.year}</p>
                </div>
                <p className="text-white/90 text-sm leading-relaxed">
                  {monument.description}
                </p>
                
                {/* AR Features */}
                <div className="space-y-2">
                  <p className="font-semibold text-xs uppercase tracking-wide">AR Features:</p>
                  <div className="flex flex-wrap gap-1">
                    {monument.arFeatures.map((feature, idx) => (
                      <span key={idx} className="bg-white/20 px-2 py-1 rounded-full text-xs">
                        {feature}
                      </span>
                    ))}
                  </div>
                </div>

                <motion.button 
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="w-full bg-white/20 backdrop-blur-sm px-4 py-3 rounded-lg font-semibold hover:bg-white/30 transition-colors"
                >
                  🥽 Start AR Experience
                </motion.button>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* AR Technology Stack */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.8 }}
        className="bg-gradient-to-r from-indigo-500 to-purple-600 text-white p-8 rounded-2xl shadow-xl"
      >
        <h3 className="text-2xl font-bold mb-6 text-center">🔧 AR Implementation Technologies</h3>
        
        <div className="grid md:grid-cols-3 gap-6">
          <div className="text-center space-y-3">
            <div className="text-4xl">📱</div>
            <h4 className="text-lg font-semibold">WebXR API</h4>
            <p className="text-indigo-100 text-sm">Browser-based AR experiences with device camera access</p>
          </div>
          
          <div className="text-center space-y-3">
            <div className="text-4xl">🎮</div>
            <h4 className="text-lg font-semibold">Three.js</h4>
            <p className="text-indigo-100 text-sm">3D graphics rendering and AR scene management</p>
          </div>
          
          <div className="text-center space-y-3">
            <div className="text-4xl">🎯</div>
            <h4 className="text-lg font-semibold">ARCore/ARKit</h4>
            <p className="text-indigo-100 text-sm">Native device AR capabilities for enhanced tracking</p>
          </div>
        </div>
      </motion.div>

      {/* Quick Kolam AR Access */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 1.0 }}
        className="bg-white p-8 rounded-2xl shadow-lg border-l-4 border-green-500"
      >
        <div className="flex items-center justify-between">
          <div className="space-y-2">
            <h3 className="text-2xl font-bold text-gray-800">🪷 Quick Kolam AR</h3>
            <p className="text-gray-600">Jump directly to Kolam pattern AR experiences</p>
          </div>
          <div className="space-x-3">
            <motion.button 
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => window.location.href = '/upload'}
              className="px-6 py-3 bg-green-500 text-white rounded-lg font-semibold hover:bg-green-600 transition-colors"
            >
              📤 Upload & AR
            </motion.button>
            <motion.button 
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => window.location.href = '/gallery'}
              className="px-6 py-3 border-2 border-green-500 text-green-600 rounded-lg font-semibold hover:bg-green-50 transition-colors"
            >
              🖼️ Gallery AR
            </motion.button>
          </div>
        </div>
      </motion.div>
    </div>
  );

  const renderMonumentExperience = () => (
    <div className="space-y-6">
      {/* Monument AR Header */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className={`bg-gradient-to-br ${selectedMonument?.color} text-white p-8 rounded-2xl shadow-xl`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-6">
            <div className="text-8xl">{selectedMonument?.icon}</div>
            <div>
              <h1 className="text-4xl font-bold mb-2">{selectedMonument?.name} AR Experience</h1>
              <p className="text-white/90 text-lg">📍 {selectedMonument?.location} • 📅 {selectedMonument?.year}</p>
              <p className="text-white/80 mt-2">{selectedMonument?.description}</p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm text-white/70">AR Status</div>
            <div className="font-semibold text-2xl">
              {arActive ? '🟢 Active' : isARSupported ? '🟡 Ready' : '🔴 Not Available'}
            </div>
          </div>
        </div>
      </motion.div>

      {/* AR Viewer Placeholder */}
      <div className="bg-white p-8 rounded-2xl shadow-lg">
        <div className="text-center space-y-6">
          <div className="text-8xl">{selectedMonument?.icon}</div>
          <h3 className="text-2xl font-bold text-gray-800">AR Experience Coming Soon</h3>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Interactive 3D monument experience with historical overlays, virtual walkthrough, 
            and immersive storytelling features will be available here.
          </p>
          
          <div className="grid md:grid-cols-3 gap-4 mt-8">
            {selectedMonument?.arFeatures.map((feature, idx) => (
              <div key={idx} className="bg-gray-50 p-4 rounded-lg">
                <div className="text-2xl mb-2">🔮</div>
                <h4 className="font-semibold text-gray-800">{feature}</h4>
                <p className="text-gray-600 text-sm mt-1">Interactive AR feature</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Back Button */}
      <div className="text-center">
        <motion.button 
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => {setCurrentView('heritage'); setSelectedMonument(null);}}
          className="px-8 py-3 bg-gray-600 text-white rounded-lg font-semibold hover:bg-gray-700 transition-colors"
        >
          ← Back to Heritage Selection
        </motion.button>
      </div>
    </div>
  );

  const renderKolamARExperience = () => (
    <div className="space-y-6">
      {/* AR Experience Header */}
      <div className="bg-gradient-to-r from-purple-500 to-indigo-600 text-white p-6 rounded-lg shadow-lg">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold mb-2">🥽 Kolam AR Viewer</h1>
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
        </div>
      )}

      {/* Back to Heritage */}
      <div className="text-center">
        <motion.button 
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => {setCurrentView('heritage'); window.history.pushState({}, '', '/ar');}}
          className="px-8 py-3 bg-gray-600 text-white rounded-lg font-semibold hover:bg-gray-700 transition-colors"
        >
          ← Back to Heritage Experience
        </motion.button>
      </div>
    </div>
  );

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      {currentView === 'heritage' && renderHeritageExperience()}
      {currentView === 'monument' && renderMonumentExperience()}
      {currentView === 'kolam' && renderKolamARExperience()}
    </motion.div>
  );
}