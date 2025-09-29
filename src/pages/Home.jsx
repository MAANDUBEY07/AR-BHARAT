import React from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";

export default function Home(){
  const features = [
    {
      icon: "🎨",
      title: "AI-Powered Generation",
      description: "Convert your images into beautiful Kolam patterns using advanced AI"
    },
    {
      icon: "📱",
      title: "4K Quality Output",
      description: "Download high-resolution 4K PNG files perfect for printing"
    },
    {
      icon: "🥽",
      title: "AR Experience",
      description: "View your patterns in augmented reality on any surface"
    }
  ];

  const stats = [
    { number: "1000+", label: "Patterns Generated" },
    { number: "4K", label: "Ultra HD Quality" },
    { number: "AR", label: "Ready Experience" },
    { number: "100%", label: "Free to Use" }
  ];

  return (
    <div className="space-y-20">
      {/* Hero Section */}
      <div className="grid lg:grid-cols-2 gap-12 items-center min-h-[60vh]">
        <motion.div 
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8 }}
        >
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-5xl lg:text-6xl font-bold bg-gradient-to-r from-orange-500 via-green-500 to-blue-500 bg-clip-text text-transparent"
          >
            AR BHARAT
          </motion.h1>
          
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="mt-6 text-xl text-gray-600 leading-relaxed"
          >
            Transform your Rangoli and Kolam images into stunning augmented reality art patterns. 
            Experience India's cultural heritage through cutting-edge AR technology.
          </motion.p>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="mt-8 flex flex-col sm:flex-row gap-4"
          >
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Link 
                to="/upload" 
                className="inline-flex items-center px-8 py-4 bg-gradient-to-r from-purple-600 to-indigo-600 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all duration-200"
              >
                <span className="mr-2">🚀</span>
                Get Started
              </Link>
            </motion.div>
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Link 
                to="/gallery" 
                className="inline-flex items-center px-8 py-4 border-2 border-gray-300 text-gray-700 font-semibold rounded-xl hover:border-indigo-500 hover:text-indigo-600 transition-all duration-200"
              >
                <span className="mr-2">🖼️</span>
                View Gallery
              </Link>
            </motion.div>
          </motion.div>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="relative"
        >
          <motion.div 
            animate={{ 
              y: [0, -10, 0],
              rotate: [0, 1, 0, -1, 0]
            }}
            transition={{ 
              duration: 6,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            className="bg-white rounded-2xl shadow-2xl p-8 border border-gray-100"
          >
            <img src="/hero-kolam.png" alt="Kolam Pattern" className="w-full rounded-lg" />
          </motion.div>
          
          {/* Floating elements */}
          <motion.div 
            animate={{ 
              y: [0, -15, 0],
              x: [0, 5, 0]
            }}
            transition={{ 
              duration: 4,
              repeat: Infinity,
              ease: "easeInOut",
              delay: 0.5
            }}
            className="absolute -top-4 -left-4 bg-purple-500 text-white p-3 rounded-full shadow-lg"
          >
            🎨
          </motion.div>
          
          <motion.div 
            animate={{ 
              y: [0, 10, 0],
              x: [0, -5, 0]
            }}
            transition={{ 
              duration: 5,
              repeat: Infinity,
              ease: "easeInOut",
              delay: 1
            }}
            className="absolute -bottom-4 -right-4 bg-indigo-500 text-white p-3 rounded-full shadow-lg"
          >
            🤖
          </motion.div>
        </motion.div>
      </div>

      {/* Stats Section */}
      <motion.div 
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.8 }}
        className="grid grid-cols-2 lg:grid-cols-4 gap-8 bg-white rounded-2xl shadow-lg p-8"
      >
        {stats.map((stat, index) => (
          <motion.div 
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 1 + index * 0.1 }}
            className="text-center"
          >
            <div className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent">
              {stat.number}
            </div>
            <div className="text-gray-600 mt-1">{stat.label}</div>
          </motion.div>
        ))}
      </motion.div>

      {/* Features Section */}
      <motion.div 
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 1.2 }}
        className="text-center"
      >
        <h2 className="text-3xl font-bold text-gray-800 mb-4">Why Choose AR BHARAT?</h2>
        <p className="text-gray-600 mb-12 max-w-2xl mx-auto">
          Experience the perfect fusion of traditional artistry and cutting-edge technology
        </p>
        
        <div className="grid md:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <motion.div 
              key={index}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 1.4 + index * 0.2 }}
              whileHover={{ y: -5 }}
              className="bg-white rounded-xl shadow-lg p-8 hover:shadow-2xl transition-all duration-300"
            >
              <div className="text-4xl mb-4">{feature.icon}</div>
              <h3 className="text-xl font-semibold text-gray-800 mb-3">{feature.title}</h3>
              <p className="text-gray-600">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* CTA Section */}
      <motion.div 
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 1.8 }}
        className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-2xl p-12 text-center"
      >
        <h2 className="text-3xl font-bold mb-4">Ready to Create Your Masterpiece?</h2>
        <p className="text-xl opacity-90 mb-8 max-w-2xl mx-auto">
          Join thousands of artists and designers who are creating stunning AR art patterns with India's cultural heritage
        </p>
        <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
          <Link 
            to="/upload"
            className="inline-flex items-center px-8 py-4 bg-white text-purple-600 font-semibold rounded-xl hover:bg-gray-100 transition-colors duration-200"
          >
            <span className="mr-2">✨</span>
            Start Creating Now
          </Link>
        </motion.div>
      </motion.div>
    </div>
  );
}